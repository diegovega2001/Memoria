"""
Módulo de modelo de visión para clustering de vehículos multi-vista.

Este módulo proporciona una clase para manejar modelos de visión computacional
pre-entrenados, adaptándolos para el clustering de vehículos con múltiples vistas
e incluyendo funcionalidades de fine-tuning y extracción de embeddings.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .Criterions import TripletLoss, ContrastiveLoss, ArcFaceLoss, ArcFaceInference
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.defaults import (
    DEFAULT_BATCH_SIZE, 
    DEFAULT_NUM_WORKERS, 
    DEFAULT_WARMUP_EPOCHS, 
    DEFAULT_TRIPLET_MARGIN, 
    DEFAULT_CONTRASTIVE_MARGIN, 
    DEFAULT_ARCFACE_SCALE, 
    DEFAULT_ARCFACE_MARGIN, 
    DEFAULT_OBJECTIVE, 
    DEFAULT_PIN_MEMORY, 
    DEFAULT_WEIGHTS,
    DEFAULT_FINETUNE_CRITERION,
    DEFAULT_FINETUNE_OPTIMIZER_TYPE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_WEIGHTS_FILENAME
)


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

SUPPORTED_MODEL_ATTRIBUTES = {'fc', 'heads', 'classifier', 'head'}

# Configuraciones conocidas de modelos
MODEL_CONFIGS = {
    'resnet': {'feature_attr': 'fc', 'feature_key': 'in_features'},
    'densenet': {'feature_attr': 'classifier', 'feature_key': 'in_features'},
    'efficientnet': {'feature_attr': 'classifier', 'feature_key': 'in_features'},
    'vit': {'feature_attr': 'heads', 'feature_key': 'in_features'},
    'swin': {'feature_attr': 'head', 'feature_key': 'in_features'},
}


class VisionModelError(Exception):
    """Excepción personalizada para errores del modelo de visión."""
    pass


class MultiViewVisionModel(nn.Module):
    """
    Modelo de visión para clasificación de vehículos multi-vista.

    Esta clase adapta modelos de visión pre-entrenados para trabajar con
    múltiples vistas de vehículos, proporcionando funcionalidades de
    fine-tuning, extracción de embeddings y clasificación.

    Attributes:
        name: Nombre descriptivo del modelo.
        model_name: Nombre del modelo base de torchvision.
        weights: Pesos pre-entrenados a utilizar.
        device: Dispositivo de cómputo (CPU/GPU).
        dataset: Dataset con las divisiones train/val/test.
        batch_size: Tamaño de batch para entrenamiento.
        model: Modelo base de torchvision.
        embedding_dim: Dimensión de los embeddings del modelo base.
        head_layer: Capa final para clasificación o metric learning.
        train_loader: DataLoader para entrenamiento.
        val_loader: DataLoader para validación.
        test_loader: DataLoader para prueba.

    Example:
        >>> model = MultiViewVisionModel(
        ...     name="ResNet50_MultiView",
        ...     model_name="resnet50",
        ...     weights="IMAGENET1K_V2",
        ...     device=torch.device("cuda"),
        ...     dataset=car_dataset,
        ...     batch_size=32
        ... )
        >>> model.finetune(criterion, optimizer, epochs=10)

    Raises:
        VisionModelError: Para errores específicos del modelo.
        ValueError: Para parámetros inválidos.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        weights: str = DEFAULT_WEIGHTS,
        device: torch.device = None,
        objective: str = DEFAULT_OBJECTIVE,
        dataset_dict: dict = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY
    ) -> None:
        """
        Inicializa el modelo de visión multi-vista.

        Args:
            name: Nombre descriptivo del modelo.
            model_name: Nombre del modelo en torchvision.models.
            weights: Identificador de pesos pre-entrenados.
            device: Dispositivo de cómputo.
            objective: Objetivo del modelo ('classification', 'metric_learning').
            dataset_dict: Diccionario con el dataset y los dataloaders con divisiones train/val/test.
            batch_size: Tamaño de batch.
            num_workers: Número de workers para DataLoaders.
            pin_memory: Si usar pin_memory en DataLoaders.

        Raises:
            VisionModelError: Si hay errores de inicialización.
            ValueError: Si los parámetros son inválidos.
        """
        super().__init__()

        # Validar parámetros
        self._validate_parameters(name, model_name, device, dataset_dict, batch_size)

        # Configuración básica
        self.name = name
        self.model_name = model_name
        self.weights = weights
        self.device = device
        self.objective = objective
        self.dataset = dataset_dict['dataset']
        self.train_loader = dataset_dict['train_loader']
        self.val_loader = dataset_dict['val_loader']
        self.test_loader = dataset_dict['test_loader']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Inicialización del modelo base
        self.model = self._initialize_base_model()
        
        # Configuración de embeddings y clasificación
        self.embedding_dim = self._extract_embedding_dimension()
        self._replace_final_layer()

        # Inicializar capa adcional según el objetivo
        self.head_layer = None
        self.arcface_layer = None
        
        if self.objective == 'classification':
            self.head_layer = self._create_classification_layer()
        elif self.objective == 'metric_learning':
            self.head_layer = self._create_embedding_projection()
            # ArcFace se inicializa solo cuando se usa como criterion, no automáticamente
        else:
            raise ValueError("objetive debe ser classifcation o metric_learning")

        # Mover a dispositivo
        self.model.to(self.device)
        self.head_layer.to(self.device)
        # arcface_layer se moverá al device cuando se cree (si es necesario)

        logging.info(f"Inicializado {self.__class__.__name__}: {self.name}")
        logging.info(f"Modelo base: {self.model_name} con pesos {self.weights}")
        logging.info(f"Objetivo: {self.objective}, Embedding dim: {self.embedding_dim}, Clases: {self.dataset.num_models}")

    def _validate_parameters(
        self,
        name: str,
        model_name: str,
        device: torch.device,
        dataset_dict: dict,
        batch_size: int,
    ) -> None:
        """Valida los parámetros de entrada."""
        if not name or not isinstance(name, str):
            raise ValueError("name debe ser una cadena no vacía")

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name debe ser una cadena no vacía")

        if not hasattr(models, 'get_model'):
            raise VisionModelError("Versión de torchvision incompatible")

        if not isinstance(device, torch.device):
            raise ValueError("device debe ser un torch.device")

        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que 0")
        
        if dataset_dict['train_loader'] is None or dataset_dict['val_loader'] is None or dataset_dict['test_loader'] is None:
            raise ValueError("Todos los DataLoaders (train_loader, val_loader, test_loader) son requeridos")
            
        if not isinstance(dataset_dict['train_loader'], DataLoader):
            raise ValueError("train_loader debe ser un DataLoader")
        if not isinstance(dataset_dict['val_loader'], DataLoader):
            raise ValueError("val_loader debe ser un DataLoader")
        if not isinstance(dataset_dict['test_loader'], DataLoader):
            raise ValueError("test_loader debe ser un DataLoader")

    def _initialize_base_model(self) -> nn.Module:
        """Inicializa el modelo base de torchvision."""
        try:
            model = models.get_model(self.model_name, weights=self.weights)
            logging.info(f"Modelo base inicializado: {self.model_name}")
            return model
        except Exception as e:
            raise VisionModelError(f"Error inicializando modelo {self.model_name}: {e}") from e

    def _extract_embedding_dimension(self) -> int:
        """
        Extrae la dimensión de embeddings del modelo base.

        Returns:
            Dimensión de los embeddings.

        Raises:
            VisionModelError: Si no se puede determinar la dimensión.
        """
        # Intentar configuraciones conocidas primero
        for model_family, config in MODEL_CONFIGS.items():
            if model_family in self.model_name.lower():
                if hasattr(self.model, config['feature_attr']):
                    layer = getattr(self.model, config['feature_attr'])
                    if hasattr(layer, config['feature_key']):
                        return getattr(layer, config['feature_key'])
                    elif hasattr(layer, 'head') and hasattr(layer.head, config['feature_key']):
                        return getattr(layer.head, config['feature_key'])

        # Búsqueda general en atributos conocidos
        for attr_name in SUPPORTED_MODEL_ATTRIBUTES:
            if hasattr(self.model, attr_name):
                layer = getattr(self.model, attr_name)
                
                # Casos comunes
                if hasattr(layer, 'in_features'):
                    return layer.in_features
                elif hasattr(layer, 'head') and hasattr(layer.head, 'in_features'):
                    return layer.head.in_features
                elif isinstance(layer, nn.Sequential) and len(layer) > 0:
                    last_layer = layer[-1]
                    if hasattr(last_layer, 'in_features'):
                        return last_layer.in_features

        raise VisionModelError(
            f"No se pudo determinar la dimensión de embeddings para {self.model_name}. "
            "Considere agregar soporte para este modelo."
        )

    def _replace_final_layer(self) -> None:
        """Reemplaza la capa final.""" 
        final_layer = nn.Identity()
        # Intentar configuraciones conocidas
        for config in MODEL_CONFIGS.values():
            if hasattr(self.model, config['feature_attr']):
                setattr(self.model, config['feature_attr'], final_layer)
                logging.debug("Capa final reemplazada según configuración del modelo")
                return

        # Búsqueda general en atributos soportados
        for attr_name in SUPPORTED_MODEL_ATTRIBUTES:
            if hasattr(self.model, attr_name):
                setattr(self.model, attr_name, final_layer)
                logging.debug("Capa final reemplazada en atributo genérico")
                return

        raise VisionModelError(f"No se pudo reemplazar la capa final para {self.model_name}") 

    def _create_classification_layer(self) -> nn.Module:
        """
        Crea la capa de clasificación final.

        Returns:
            Capa linear para clasificación.
        """
        # Dimensión de entrada: embedding_dim * número de vistas
        input_dim = self.embedding_dim * self.dataset.num_views
        output_dim = self.dataset.num_models

        classification_layer = nn.Linear(input_dim, output_dim)
        
        # Inicialización Xavier
        nn.init.xavier_uniform_(classification_layer.weight)
        nn.init.zeros_(classification_layer.bias)

        logging.debug(f"Capa de clasificación creada: {input_dim} → {output_dim}")
        return classification_layer

    def _create_embedding_projection(self) -> nn.Module:
        """
        Crea una capa de proyección para ajustar el tamaño de embeddings.
        
        Returns:
            Capa de proyección para embeddings.
        """
        input_dim = self.embedding_dim * self.dataset.num_views
        output_dim = self.embedding_dim
        
        projection_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),            
            nn.Dropout(p=0.3),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # Inicialización
        for layer in projection_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        logging.debug(f"Capa de proyección creada: {input_dim} → {output_dim}")
        return projection_layer

    def create_arcface_layer(self, scale: float = DEFAULT_ARCFACE_SCALE, margin: float = DEFAULT_ARCFACE_MARGIN) -> ArcFaceLoss:
        """
        Crea una capa ArcFace para metric learning
        
        Args:
            scale: Factor de escala para ArcFace.
            margin: Margen angular para ArcFace.
            
        Returns:
            Instancia de ArcFaceLoss configurada.
        """
        
        arcface_layer = ArcFaceLoss(
            embedding_dim=self.embedding_dim,
            num_classes=self.dataset.num_models,
            scale=scale,
            margin=margin
        )
        logging.debug(f"Capa ArcFace creada: dim={self.embedding_dim}, clases={self.dataset.num_models}")
        return arcface_layer
    
    def create_triplet_loss(self, margin: float = DEFAULT_TRIPLET_MARGIN) -> TripletLoss:
        """
        Crea una función de pérdida Triplet.
        
        Args:
            margin: Margen para Triplet Loss.
            
        Returns:
            Instancia de TripletLoss configurada.
        """
        return TripletLoss(margin=margin)

    def create_contrastive_loss(self, margin: float = DEFAULT_CONTRASTIVE_MARGIN) -> ContrastiveLoss:
        """
        Crea una función de pérdida Contrastive.
        
        Args:
            margin: Margen para Contrastive Loss.
            
        Returns:
            Instancia de ContrastiveLoss configurada.
        """
        return ContrastiveLoss(margin=margin)


    def extract_embeddings(
        self,
        dataloader: DataLoader,
        apply_scaling: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Extrae embeddings de un DataLoader.

        Args:
            dataloader: DataLoader con las imágenes.
            apply_scaling: Si aplicar StandardScaler a los embeddings.
            show_progress: Si mostrar barra de progreso.

        Returns:
            Tensor con los embeddings extraídos.

        Raises:
            VisionModelError: Si hay errores durante la extracción.
        """
        try:
            self.model.eval()
            embeddings = []

            with torch.no_grad():
                iterator = tqdm(dataloader, desc='Extracting embeddings', leave=False) if show_progress else dataloader
                
                for batch in iterator:
                    images = batch['images']
                    
                    # Procesar el batch completo (no imagen por imagen)
                    batch_embeddings = torch.flatten(
                        self.model(images.to(self.device)), 
                        start_dim=1
                    )
                    
                    embeddings.append(batch_embeddings.cpu())

            # Concatenar todos los embeddings
            all_embeddings = torch.cat(embeddings, dim=0)

            # Aplicar escalado si se solicita
            if apply_scaling:
                scaler = StandardScaler()
                scaled_embeddings = scaler.fit_transform(all_embeddings.numpy())
                return torch.tensor(scaled_embeddings, dtype=torch.float32)
            
            return all_embeddings

        except Exception as e:
            raise VisionModelError(f"Error extrayendo embeddings: {e}") from e

    def extract_val_embeddings(self, apply_scaling: bool = True) -> torch.Tensor:
        """
        Extrae embeddings del conjunto de validación.

        Args:
            apply_scaling: Si aplicar escalado a los embeddings.

        Returns:
            Embeddings del conjunto de validación.
        """
        return self.extract_embeddings(
            self.val_loader, 
            apply_scaling=apply_scaling,
            show_progress=True
        )
    
    def extract_test_embeddings(self, apply_scaling: bool = True) -> torch.Tensor:
        """
        Extrae embeddings del conjunto de prueba.

        Args:
            apply_scaling: Si aplicar escalado a los embeddings.

        Returns:
            Embeddings del conjunto de prueba.
        """
        return self.extract_embeddings(
            self.test_loader, 
            apply_scaling=apply_scaling,
            show_progress=True
        )

    def finetune(
        self,
        criterion: nn.Module = DEFAULT_FINETUNE_CRITERION,
        optimizer: torch.optim.Optimizer = DEFAULT_FINETUNE_OPTIMIZER_TYPE,
        epochs: int = DEFAULT_FINETUNE_EPOCHS,
        warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, List[float]]:
        """
        Realiza fine-tuning del modelo.

        Args:
            criterion: Función de pérdida.
            optimizer: Optimizador.
            epochs: Número de épocas.
            warmup_epochs: Épocas de warmup para el learning rate.
            scheduler: Scheduler de learning rate personalizado.
            early_stopping: Configuración de early stopping.
            save_best: Si guardar el mejor modelo durante entrenamiento.
            checkpoint_dir: Directorio para guardar checkpoints.

        Returns:
            Diccionario con historial de entrenamiento.

        Raises:
            VisionModelError: Si hay errores durante el entrenamiento.
        """
        try:
            # Inicializar arcface_layer solo si el criterion es ArcFaceLoss y aún no existe
            if isinstance(criterion, ArcFaceLoss) and self.arcface_layer is None:
                self.arcface_layer = self.create_arcface_layer()
                self.arcface_layer.to(self.device)
                logging.info("ArcFace layer inicializada para entrenamiento")
            
            # Configuración de entrenamiento
            history = {
                'train_loss': [], 
                'val_loss': [], 
                'val_accuracy': [],
                'val_recall@1': [],
                'val_recall@5': []
            }

            # Para metric learning con Triplet/Contrastive, usamos Recall@1 en vez de accuracy
            use_recall_for_tracking = (self.objective == 'metric_learning' and 
                                       not isinstance(criterion, ArcFaceLoss))
            
            best_val_metric = 0.0
            patience_counter = 0

            # Scheduler de warmup si se especifica
            warmup_scheduler = None
            if warmup_epochs > 0:
                warmup_scheduler = self._create_warmup_scheduler(optimizer, warmup_epochs)

            # Configuración de early stopping
            early_stop_patience = early_stopping.get('patience', 10) if early_stopping else None
            early_stop_min_delta = early_stopping.get('min_delta', 0.001) if early_stopping else None

            logging.info(f"Iniciando fine-tuning: {epochs} épocas, warmup: {warmup_epochs}")

            for epoch in range(epochs):
                # Entrenamiento
                train_loss = self._train_epoch(criterion, optimizer, epoch, epochs)
                
                # Warmup scheduler
                if warmup_scheduler and epoch < warmup_epochs:
                    warmup_scheduler.step()
                elif scheduler:
                    scheduler.step()

                # Validación
                val_loss, val_accuracy, val_recalls = self._validate_epoch(criterion, epoch, epochs)

                # Actualizar historial
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                history['val_recall@1'].append(val_recalls[1] if val_recalls else None)
                history['val_recall@5'].append(val_recalls[5] if val_recalls else None)

                # Determinar métrica de seguimiento según el tipo de entrenamiento
                if use_recall_for_tracking and val_recalls:
                    current_metric = val_recalls[1]  # Usar Recall@1
                    metric_name = 'Recall@1'
                else:
                    current_metric = val_accuracy  # Usar accuracy
                    metric_name = 'Val Acc'

                # Log de progreso
                log_msg = (
                    f'Epoch {epoch+1}/{epochs} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f}'
                )
                
                # Solo mostrar accuracy si es significativa (classification o ArcFace)
                if self.objective == 'classification' or isinstance(criterion, ArcFaceLoss):
                    log_msg += f' | Val Acc: {val_accuracy:.2f}%'
                
                # Mostrar Recall@K para metric learning
                if val_recalls:
                    log_msg += f" | Recall@1: {val_recalls[1]:.2f}% | Recall@5: {val_recalls[5]:.2f}%"
                
                logging.info(log_msg)

                # Guardar mejor modelo
                if save_best and current_metric > best_val_metric:
                    best_val_metric = current_metric
                    if checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, epoch, current_metric, 'best')

                # Early stopping
                if early_stopping:
                    if current_metric > best_val_metric - early_stop_min_delta:
                        patience_counter = 0
                        best_val_metric = current_metric
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stop_patience:
                        logging.info(f"Early stopping triggered at epoch {epoch+1} based on {metric_name}")
                        break

            logging.info(f"Fine-tuning completado.")
            return history

        except Exception as e:
            raise VisionModelError(f"Error durante fine-tuning: {e}") from e

    def _create_warmup_scheduler(
        self, 
        optimizer: torch.optim.Optimizer, 
        warmup_epochs: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Crea scheduler de warmup para learning rate."""
        def lr_lambda(current_epoch: int) -> float:
            return float(current_epoch + 1) / warmup_epochs if current_epoch < warmup_epochs else 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _compute_triplet_loss(self, criterion: TripletLoss, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calcula Triplet Loss generando triplets de forma online.
        
        Args:
            criterion: Función de pérdida TripletLoss.
            embeddings: Embeddings del batch.
            labels: Etiquetas del batch.
            
        Returns:
            Pérdida triplet calculada.
        """
        batch_size = embeddings.size(0)
        if batch_size < 3:
            # Fallback para batches muy pequeños
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Generar triplets de forma online (estrategia semi-hard)
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Encontrar índices de positivos y negativos
            pos_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
            neg_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 1 and len(neg_indices) > 0:
                # Seleccionar positivo (diferente del ancla)
                pos_candidates = pos_indices[pos_indices != i]
                if len(pos_candidates) > 0:
                    pos_idx = pos_candidates[torch.randint(len(pos_candidates), (1,))].item()
                    neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))].item()
                    
                    anchors.append(embeddings[i])
                    positives.append(embeddings[pos_idx])
                    negatives.append(embeddings[neg_idx])
        
        if len(anchors) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        anchor_batch = torch.stack(anchors)
        positive_batch = torch.stack(positives)
        negative_batch = torch.stack(negatives)
        
        return criterion(anchor_batch, positive_batch, negative_batch)

    def _compute_contrastive_loss(self, criterion: ContrastiveLoss, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calcula Contrastive Loss generando pares de forma online.
        
        Args:
            criterion: Función de pérdida ContrastiveLoss.
            embeddings: Embeddings del batch.
            labels: Etiquetas del batch.
            
        Returns:
            Pérdida contrastiva calculada.
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Generar pares de forma online
        embeddings1, embeddings2, pair_labels = [], [], []
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                embeddings1.append(embeddings[i])
                embeddings2.append(embeddings[j])
                # Label 0 para misma clase, 1 para diferentes clases
                pair_labels.append(0 if labels[i] == labels[j] else 1)
        
        if len(embeddings1) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        emb1_batch = torch.stack(embeddings1)
        emb2_batch = torch.stack(embeddings2)
        label_batch = torch.tensor(pair_labels, device=embeddings.device, dtype=torch.float)
        
        return criterion(emb1_batch, emb2_batch, label_batch)

    def _predict_by_similarity(self, embeddings: torch.Tensor, dataloader: DataLoader) -> torch.Tensor:
        """
        Predice clases usando similitud con centroides de clase.
        
        Args:
            embeddings: Embeddings a predecir.
            dataloader: DataLoader completo para calcular centroides de clase.
            
        Returns:
            Predicciones basadas en similitud con centroides.
        """
        # Calcular centroides de clase desde el dataloader completo
        class_embeddings = {}
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                labels = batch['labels']
                
                # Extraer embeddings
                batch_embeddings = torch.flatten(
                    self.model(images), 
                    start_dim=1
                )
                
                # Proyectar si es metric_learning
                if self.objective == 'metric_learning':
                    batch_embeddings = self.head_layer(batch_embeddings)
                
                batch_embeddings = batch_embeddings.cpu()
                
                # Acumular por clase
                for emb, label in zip(batch_embeddings, labels):
                    label_item = label.item()
                    if label_item not in class_embeddings:
                        class_embeddings[label_item] = []
                    class_embeddings[label_item].append(emb)
        
        # Calcular centroides
        centroids = {}
        for label, emb_list in class_embeddings.items():
            centroids[label] = torch.stack(emb_list).mean(dim=0)
        
        # Crear matriz de centroides ordenada por clase
        sorted_labels = sorted(centroids.keys())
        centroid_matrix = torch.stack([centroids[label] for label in sorted_labels])
        
        # Normalizar para cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        centroid_matrix_norm = F.normalize(centroid_matrix, p=2, dim=1)
        
        # Calcular similitud
        similarity = torch.mm(embeddings_norm, centroid_matrix_norm.t())
        
        # Predecir clase con mayor similitud
        _, predicted_indices = similarity.max(dim=1)
        predictions = torch.tensor([sorted_labels[idx] for idx in predicted_indices])
        
        return predictions
    
    def _recall_at_k(self, embeddings: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> Dict[int, float]:
        """
        Calcula Recall@K para embeddings de validación.

        Args:
            embeddings: Tensor (N, D) con embeddings normalizados.
            labels: Tensor (N,) con etiquetas.
            ks: Tuplas de valores de K a evaluar.

        Returns:
            Diccionario {K: recall@K}
        """
        recalls = {}
        embeddings = F.normalize(embeddings, p=2, dim=1)  
        sim_matrix = torch.matmul(embeddings, embeddings.t())

        # Evitar auto-similitud
        sim_matrix.fill_diagonal_(-float("inf"))

        # Para cada fila, obtener topK vecinos
        for k in ks:
            topk = sim_matrix.topk(k, dim=1).indices  # (N, K)
            topk_labels = labels[topk]  # Etiquetas de los vecinos

            # Éxito si al menos 1 vecino comparte clase
            correct = (topk_labels == labels.unsqueeze(1)).any(dim=1).float()
            recalls[k] = correct.mean().item() * 100  # En %

        return recalls

    def _train_epoch(
        self,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int
    ) -> float:
        """Ejecuta una época de entrenamiento."""
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', leave=False) as pbar:
            for batch in pbar:
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                # Forward pass - extraer embeddings
                embeddings = torch.flatten(
                    self.model(images.to(self.device)), 
                    start_dim=1
                )  
                
                # Calcular pérdida según el objetivo
                if self.objective == 'classification':
                    outputs = self.head_layer(embeddings)
                    loss = criterion(outputs, labels)
                    
                elif self.objective == 'metric_learning':
                    # Proyectar embeddings a través de head_layer
                    projected_embeddings = self.head_layer(embeddings)
                    
                    if isinstance(criterion, ArcFaceLoss):
                        # Para ArcFace, usar la capa arcface_layer
                        logits = self.arcface_layer(projected_embeddings, labels)
                        loss = F.cross_entropy(logits, labels)

                    elif isinstance(criterion, TripletLoss):
                        # Para Triplet Loss, necesitamos crear triplets
                        loss = self._compute_triplet_loss(criterion, projected_embeddings, labels)
                        
                    elif isinstance(criterion, ContrastiveLoss):
                        # Para Contrastive Loss, necesitamos crear pares
                        loss = self._compute_contrastive_loss(criterion, projected_embeddings, labels)
                        
                    else:
                        raise VisionModelError(f"Función de pérdida no soportada para metric_learning.")
                else:
                    raise VisionModelError(f"Objetivo '{self.objective}' no reconocido.")

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def _validate_epoch(
        self, 
        criterion: nn.Module, 
        epoch: int, 
        total_epochs: int,
    ) -> Tuple[float, float]:
        """Ejecuta una época de validación."""
        self.model.eval()
        if self.head_layer is not None:
            self.head_layer.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            with tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]', leave=False) as pbar:
                for batch in pbar:
                    images = batch['images'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # Forward pass - extraer embeddings
                    embeddings = torch.flatten(
                        self.model(images.to(self.device)), 
                        start_dim=1
                    )
                    
                    # Calcular pérdida y accuracy según el objetivo
                    if self.objective == 'classification':
                        outputs = self.head_layer(embeddings)
                        loss = criterion(outputs, labels)
                        _, predicted = outputs.max(1)
                        # Guardar embeddings sin proyección para recalls
                        all_embeddings.append(embeddings.cpu())
                        all_labels.append(labels.cpu())

                    elif self.objective == 'metric_learning':
                        # Proyectar embeddings a través de head_layer
                        projected_embeddings = self.head_layer(embeddings)
                        
                        if isinstance(criterion, ArcFaceLoss):
                            logits = self.arcface_layer(projected_embeddings, labels)
                            loss = F.cross_entropy(logits, labels)
                            
                            # Para accuracy, usar ArcFaceInference (sin margen)
                            arcface_inf = ArcFaceInference(
                                weight=self.arcface_layer.weight,
                                scale=self.arcface_layer.scale
                            )
                            inference_logits = arcface_inf(projected_embeddings)
                            _, predicted = inference_logits.max(1)
                            
                        elif isinstance(criterion, TripletLoss):
                            loss = self._compute_triplet_loss(criterion, projected_embeddings, labels)
                            # Para triplet/contrastive, no calculamos accuracy tradicional
                            # La métrica real es Recall@K que se calcula al final
                            # Usamos un placeholder que no afecta las métricas finales
                            predicted = torch.zeros_like(labels)  # Placeholder para no inflar accuracy

                        elif isinstance(criterion, ContrastiveLoss):
                            loss = self._compute_contrastive_loss(criterion, projected_embeddings, labels)
                            # Para triplet/contrastive, no calculamos accuracy tradicional
                            # La métrica real es Recall@K que se calcula al final
                            # Usamos un placeholder que no afecta las métricas finales
                            predicted = torch.zeros_like(labels)  # Placeholder para no inflar accuracy
                            
                        else:
                            raise VisionModelError(f"Función de pérdida no soportada para metric_learning.")
                        
                        # Guardar embeddings proyectados para recalls
                        all_embeddings.append(projected_embeddings.cpu())
                        all_labels.append(labels.cpu())
                    
                    else:
                        raise VisionModelError(f"Objetivo '{self.objective}' no reconocido.")

                    total_loss += loss.item()
                    
                    # Accuracy
                    total_samples += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    current_acc = 100 * correct / total_samples
                    pbar.set_postfix({'acc': f'{current_acc:.2f}%'})

        avg_loss = total_loss / len(self.val_loader)
        
        # Para metric learning (excepto ArcFace), la accuracy no es significativa
        # ya que no usamos clasificación directa. La métrica real es Recall@K
        if self.objective == 'metric_learning' and not isinstance(criterion, ArcFaceLoss):
            accuracy = 0.0  # No significativa para Triplet/Contrastive
        else:
            accuracy = 100 * correct / total_samples
        
        recalls = {}
        if self.objective == 'metric_learning':
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            recalls = self._recall_at_k(all_embeddings, all_labels, ks=(1, 5))

        return avg_loss, accuracy, recalls

    def evaluate(self, dataloader: Optional[DataLoader] = None, use_similarity: bool = None) -> Dict[str, float]:
        """
        Evalúa el modelo en un conjunto de datos.

        Args:
            dataloader: DataLoader a evaluar. Si None, usa val_loader.
            use_similarity: Si usar predicción por similitud. Si None, se decide automáticamente.

        Returns:
            Diccionario con métricas de evaluación.
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        if self.head_layer is not None:
            self.head_layer.eval()
        if self.arcface_layer is not None:
            self.arcface_layer.eval()

        all_embeddings = []
        all_labels = []

        # Decidir método de predicción
        if use_similarity is None:
            # Si tenemos ArcFace, usar predicción basada en logits (no similitud)
            # De lo contrario, decidir por objetivo
            use_similarity = (self.arcface_layer is None) and (self.objective == 'metric_learning')

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Extraer embeddings
                embeddings = torch.flatten(
                    self.model(images.to(self.device)), 
                    start_dim=1
                )
                
                # Si es metric_learning, proyectar embeddings
                if self.objective == 'metric_learning':
                    embeddings = self.head_layer(embeddings)

                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Realizar predicción según el método
            if use_similarity:
                # Metric learning sin ArcFace: usar predicción por similitud con centroides
                predicted = self._predict_by_similarity(all_embeddings, dataloader)
                accuracy = 100.0 * (predicted == all_labels).float().mean().item()
                
                return {
                    'accuracy': accuracy,
                    'total_samples': len(all_labels),
                    'predictions': predicted.numpy(),
                    'labels': all_labels.numpy(),
                    'method': 'similarity_centroids'
                }
            else:
                # Clasificación o ArcFace: usar head_layer o arcface inference
                if self.arcface_layer is not None:
                    # Usar ArcFaceInference para predicción sin labels
                    arcface_inf = ArcFaceInference(
                        weight=self.arcface_layer.weight,
                        scale=self.arcface_layer.scale
                    )
                    logits = arcface_inf(all_embeddings.to(self.device))
                    _, predicted = logits.max(1)
                    predicted = predicted.cpu()
                else:
                    # Clasificación estándar
                    outputs = self.head_layer(all_embeddings.to(self.device))
                    _, predicted = outputs.max(1)
                    predicted = predicted.cpu()

                correct = (predicted == all_labels).sum().item()
                accuracy = 100 * correct / len(all_labels) if len(all_labels) > 0 else 0.0
                return {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': len(all_labels),
                    'predictions': predicted.numpy(),
                    'labels': all_labels.numpy(),
                    'method': 'classification'
                }


    def save_model(self, save_path: Union[str, Path], **kwargs) -> None:
        """
        Guarda el modelo completo.

        Args:
            save_path: Directorio donde guardar el modelo.
            **kwargs: Metadatos adicionales a guardar.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / DEFAULT_WEIGHTS_FILENAME

        # Preparar checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'head_layer_state_dict': self.head_layer.state_dict(),
            'model_config': {
                'name': self.name,
                'model_name': self.model_name,
                'weights': self.weights,
                'embedding_dim': self.embedding_dim,
                'num_classes': self.dataset.num_models,
                'num_views': self.dataset.num_views
            },
            'metadata': kwargs
        }

        torch.save(checkpoint, model_path)
        logging.info(f"Modelo guardado en: {model_path}")

    def load_model(self, load_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Carga un modelo guardado.

        Args:
            load_path: Directorio donde está el modelo.

        Returns:
            Metadatos del modelo cargado.

        Raises:
            VisionModelError: Si no se puede cargar el modelo.
        """
        load_path = Path(load_path)
        model_path = load_path / DEFAULT_WEIGHTS_FILENAME

        if not model_path.exists():
            raise VisionModelError(f"No se encontró modelo en: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Cargar estados
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.head_layer.load_state_dict(checkpoint['head_layer_state_dict'])
            
            # Verificar compatibilidad
            config = checkpoint.get('model_config', {})
            if config.get('num_classes') != self.dataset.num_models:
                logging.warning(
                    f"Número de clases no coincide: {config.get('num_classes')} vs {self.dataset.num_models}"
                )

            logging.info(f"Modelo cargado desde: {model_path}")
            return checkpoint.get('metadata', {})

        except Exception as e:
            raise VisionModelError(f"Error cargando modelo: {e}") from e

    def _save_checkpoint(
        self, 
        checkpoint_dir: Union[str, Path], 
        epoch: int, 
        accuracy: float, 
        suffix: str = ''
    ) -> None:
        """Guarda un checkpoint durante el entrenamiento."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"checkpoint_epoch_{epoch}_{suffix}.pth" if suffix else f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'head_layer_state_dict': self.head_layer.state_dict(),
            'accuracy': accuracy,
            'model_config': {
                'name': self.name,
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim
            }
        }

        torch.save(checkpoint, checkpoint_path)

    def forward(self, images: Union[List, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass del modelo.

        Args:
            images: Imágenes de entrada en formato lista o tensor.

        Returns:
            Embeddings o logits según el contexto.

        Raises:
            ValueError: Si el formato de entrada no es soportado.
        """
        self.model.eval()

        with torch.no_grad():
            if isinstance(images, list):
                if len(images) > 0 and isinstance(images[0], list):
                    # Batch de muestras multi-vista
                    batch_embeddings = []
                    for views in images:
                        sample_emb = torch.cat([
                            torch.flatten(
                                self.model(img.unsqueeze(0).to(self.device)), 
                                start_dim=1
                            )
                            for img in views
                        ], dim=1)
                        batch_embeddings.append(sample_emb)
                    return torch.cat(batch_embeddings, dim=0)
                else:
                    # Muestra individual multi-vista
                    return torch.cat([
                        torch.flatten(
                            self.model(img.unsqueeze(0).to(self.device)), 
                            start_dim=1
                        )
                        for img in images
                    ], dim=1)
            elif isinstance(images, torch.Tensor):
                # Tensor directo
                return self.model(images.to(self.device))
            else:
                raise ValueError(f"Formato de entrada no soportado: {type(images)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada del modelo.

        Returns:
            Diccionario con información del modelo.
        """
        return {
            'name': self.name,
            'model_name': self.model_name,
            'weights': self.weights,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.dataset.num_models,
            'num_views': self.dataset.num_views,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 ** 2)
        }

    def __str__(self) -> str:
        """Representación string del modelo."""
        info = self.get_model_info()
        return (
            f"MultiViewVisionModel(\n"
            f"  name='{info['name']}',\n"
            f"  base_model='{info['model_name']}',\n"
            f"  device='{info['device']}',\n"
            f"  classes={info['num_classes']},\n"
            f"  views={info['num_views']},\n"
            f"  batch_size={info['batch_size']},\n"
            f"  parameters={info['total_parameters']:,}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Representación concisa para debugging."""
        return f"MultiViewVisionModel(name='{self.name}', model='{self.model_name}')"


# Funciones de conveniencia
def create_vision_model(
    model_name: str,
    dataset_dict: dict,
    device: Optional[torch.device] = None,
    weights: str = 'DEFAULT',
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs
) -> MultiViewVisionModel:
    """
    Función de conveniencia para crear un modelo de visión.

    Args:
        model_name: Nombre del modelo base.
        dataset_dict: Diccionario con dataset y dataloaders.
        device: Dispositivo de cómputo.
        weights: Pesos pre-entrenados.
        batch_size: Tamaño de batch.
        **kwargs: Argumentos adicionales.

    Returns:
        Modelo de visión inicializado.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    name = kwargs.pop('name', f"{model_name.title()}_MultiView")

    return MultiViewVisionModel(
        name=name,
        model_name=model_name,
        weights=weights,
        device=device,
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        **kwargs
    )