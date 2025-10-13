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

from .Criterions import TripletLoss, ContrastiveLoss, ArcFaceLayer, ArcFaceInference
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
    DEFAULT_OBJECTIVE, 
    DEFAULT_PIN_MEMORY, 
    DEFAULT_WEIGHTS,
    DEFAULT_FINETUNE_CRITERION,
    DEFAULT_FINETUNE_OPTIMIZER_TYPE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_WEIGHTS_FILENAME,
    DEFAULT_OUTPUT_EMBEDDING_DIM,
    DEFAULT_USE_AMP,
    SUPPORTED_MODEL_ATTRIBUTES,
    MODEL_CONFIGS
)


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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
        head_layer: Capa final para clasificación, metric learning o ArcFace.
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
        
        if self.objective == 'classification':
            self.head_layer = self._create_classification_layer()
        elif self.objective == 'metric_learning':
            self.head_layer = self._create_embedding_projection()
        elif self.objective == 'ArcFace':
            self.head_layer = nn.Sequential(self._create_embedding_projection(),
                                            self._create_arcface_layer())
        else:
            raise ValueError("objetive debe ser classifcation o metric_learning")

        # Mover a dispositivo
        self.model.to(self.device)
        self.head_layer.to(self.device)

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
        
        # Creación de la capa de clasificación
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
        # Dimensión de entrada: embedding_dim * número de vistas
        input_dim = self.embedding_dim * self.dataset.num_views
        output_dim = DEFAULT_OUTPUT_EMBEDDING_DIM
        
        # Creación de la capa de proyección de embeddings 
        projection_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),            
            nn.Dropout(p=0.3),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # Inicialización Xavier
        for layer in projection_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        logging.debug(f"Capa de proyección creada: {input_dim} → {output_dim}")
        return projection_layer

    def _create_arcface_layer(self) -> ArcFaceLayer:
        """
        Crea una capa ArcFace para metric learning
        
        Args:
            scale: Factor de escala para ArcFace.
            margin: Margen angular para ArcFace.
            
        Returns:
            Instancia de ArcFaceLayer configurada.
        """
        
        # Creación de la capa ArcFace
        arcface_layer = ArcFaceLayer(
            embedding_dim=DEFAULT_OUTPUT_EMBEDDING_DIM,
            num_classes=self.dataset.num_models,
        )
        logging.debug(f"Capa ArcFace creada: dim={DEFAULT_OUTPUT_EMBEDDING_DIM}, clases={self.dataset.num_models}")
        return arcface_layer

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
                    images = batch['images'].to(self.device)

                    # Usar el método forward
                    batch_embeddings = self.forward(images)
                    embeddings.append(batch_embeddings.cpu())

            # Concatenar todos los embeddings
            all_embeddings = torch.cat(embeddings, dim=0)

            # Aplicar escalado 
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
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = DEFAULT_USE_AMP,
        gradient_clip_value: Optional[float] = None
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
            use_amp: Si usar Automatic Mixed Precision para acelerar entrenamiento.
            gradient_clip_value: Valor máximo para gradient clipping. None para no aplicar.

        Returns:
            Diccionario con historial de entrenamiento.

        Raises:
            VisionModelError: Si hay errores durante el entrenamiento.
        """
        try:
            # Configuración de entrenamiento
            history = {
                'train_loss': [], 
                'val_loss': [], 
                'val_accuracy': [],
                'val_recall@1': [],
                'val_recall@5': []
            }
            
            best_val_loss = float('inf')
            patience_counter = 0

            # Scheduler de warmup si se especifica
            warmup_scheduler = None
            if warmup_epochs > 0:
                warmup_scheduler = self._create_warmup_scheduler(optimizer, warmup_epochs)

            # Configuración de early stopping
            early_stop_patience = early_stopping.get('patience', 10) if early_stopping else None
            early_stop_min_delta = early_stopping.get('min_delta', 0.001) if early_stopping else None

            # Configurar GradScaler para AMP si está habilitado y hay GPU disponible
            scaler = None
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            if use_amp and device_type == 'cuda':
                scaler = torch.amp.GradScaler('cuda')
                logging.info("AMP (Automatic Mixed Precision) habilitado con GradScaler")
            elif use_amp and device_type != 'cuda':
                logging.warning("AMP solicitado pero no hay GPU disponible, deshabilitando AMP")
                use_amp = False

            logging.info(f"Iniciando fine-tuning: {epochs} épocas, warmup: {warmup_epochs}, AMP: {use_amp}")

            for epoch in range(epochs):
                # Entrenamiento
                train_loss = self._train_epoch(
                    criterion, optimizer, epoch, epochs, 
                    use_amp=use_amp, scaler=scaler,
                    gradient_clip_value=gradient_clip_value
                )
                
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

                # Log de progreso
                log_msg = (
                    f'Epoch {epoch+1}/{epochs} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f}'
                )
                
                # Solo mostrar accuracy si es significativa (classification o ArcFace)
                if self.objective in ['classification', 'ArcFace']:
                    log_msg += f' | Val Acc: {val_accuracy:.2f}%'
                
                # Mostrar Recall@K para metric learning
                if val_recalls:
                    log_msg += f" | Recall@1: {val_recalls[1]:.2f}% | Recall@5: {val_recalls[5]:.2f}%"
                
                logging.info(log_msg)

                # Guardar mejor modelo
                if save_best and val_loss < best_val_loss:
                    if checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, epoch, val_loss, 'best')

                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss - early_stop_min_delta:
                        patience_counter = 0
                        best_val_loss = val_loss
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stop_patience:
                        logging.info(f"Early stopping triggered at epoch {epoch+1} based on validaton loss")
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
    
    def _predict_by_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predice clases usando similitud con centroides de clase.
        
        Args:
            embeddings: Embeddings a predecir (del val/test loader).
            
        Returns:
            Predicciones basadas en similitud con centroides del conjunto de entrenamiento.
        """
        # Calcular centroides de clase desde el train loader
        class_embeddings = {}
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.train_loader:
                images = batch['images'].to(self.device)
                labels = batch['labels']
                
                # Usar el método forward que maneja múltiples vistas correctamente
                batch_embeddings = self.forward(images)
                
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
        total_epochs: int,
        use_amp: bool = False,
        scaler: Optional[torch.amp.GradScaler] = None,
        gradient_clip_value: Optional[float] = None
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

                with torch.amp.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=use_amp):
                    # Forward pass - extraer embeddings
                    # Manejar múltiples vistas
                    if images.ndim == 4:
                        images = images.unsqueeze(1)
                    
                    B, V, C, H, W = images.shape
                    images_reshaped = images.view(B * V, C, H, W)
                    features = self.model(images_reshaped)
                    features = features.view(B, V, -1)
                    # Concatenar vistas (no promediar) para preservar información
                    embeddings = features.flatten(start_dim=1)  # [B, V*embedding_dim]  
                
                    # Calcular pérdida y accuracy según el objetivo
                    if self.objective == 'classification':
                        outputs = self.head_layer(embeddings)
                        loss = criterion(outputs, labels)
                    elif self.objective == 'ArcFace':
                        projection_layer, arcface_layer = self.head_layer
                        embeddings = projection_layer(embeddings)
                        logits = arcface_layer(embeddings, labels)
                        loss = criterion(logits, labels)               
                    elif self.objective == 'metric_learning':
                        # Proyectar embeddings a través de head_layer
                        projected_embeddings = self.head_layer(embeddings)
                
                        if isinstance(criterion, TripletLoss):
                            # Para TripletLoss, necesitamos hacer hard negative mining
                            # Crear tripletas (anchor, positive, negative) del batch
                            batch_size = projected_embeddings.size(0)
                            
                            # Calcular matriz de distancias para mining
                            dist_matrix = torch.cdist(projected_embeddings, projected_embeddings, p=2)
                            
                            anchors, positives, negatives = [], [], []
                            for i in range(batch_size):
                                # Mask de muestras de la misma clase (excluyendo el anchor mismo)
                                pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
                                # Mask de muestras de diferente clase
                                neg_mask = labels != labels[i]
                                
                                if pos_mask.any() and neg_mask.any():
                                    # Hard positive: el más lejano de la misma clase
                                    hardest_pos_idx = dist_matrix[i][pos_mask].argmax()
                                    pos_idx = torch.where(pos_mask)[0][hardest_pos_idx]
                                    
                                    # Hard negative: el más cercano de diferente clase
                                    hardest_neg_idx = dist_matrix[i][neg_mask].argmin()
                                    neg_idx = torch.where(neg_mask)[0][hardest_neg_idx]
                                    
                                    anchors.append(projected_embeddings[i])
                                    positives.append(projected_embeddings[pos_idx])
                                    negatives.append(projected_embeddings[neg_idx])
                            
                            if len(anchors) > 0:
                                anchor_batch = torch.stack(anchors)
                                positive_batch = torch.stack(positives)
                                negative_batch = torch.stack(negatives)
                                # Usar el forward del criterion directamente
                                loss = criterion(anchor_batch, positive_batch, negative_batch)
                            else:
                                # Si no hay tripletas válidas en este batch
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        
                        elif isinstance(criterion, ContrastiveLoss):
                            # Para ContrastiveLoss (modo tradicional), crear pares
                            batch_size = projected_embeddings.size(0)
                            pairs_1, pairs_2, pair_labels = [], [], []
                            
                            for i in range(batch_size):
                                for j in range(i + 1, batch_size):
                                    pairs_1.append(projected_embeddings[i])
                                    pairs_2.append(projected_embeddings[j])
                                    # Label: 0 si misma clase, 1 si diferente clase
                                    pair_labels.append(0 if labels[i] == labels[j] else 1)
                            
                            if len(pairs_1) > 0:
                                pairs_1_batch = torch.stack(pairs_1)
                                pairs_2_batch = torch.stack(pairs_2)
                                pair_labels_batch = torch.tensor(pair_labels, device=self.device, dtype=torch.float)
                                # Usar el forward del criterion directamente
                                loss = criterion(pairs_1_batch, pairs_2_batch, pair_labels_batch)
                            else:
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        
                        else:
                            raise VisionModelError(f"Función de pérdida no soportada para metric_learning.")
                    else:
                        raise VisionModelError(f"Objetivo '{self.objective}' no reconocido.")

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if gradient_clip_value is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
                        torch.nn.utils.clip_grad_norm_(self.head_layer.parameters(), gradient_clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
                        torch.nn.utils.clip_grad_norm_(self.head_layer.parameters(), gradient_clip_value)
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
                    # Manejar múltiples vistas
                    if images.ndim == 4:
                        images = images.unsqueeze(1)
                    
                    B, V, C, H, W = images.shape
                    images_reshaped = images.view(B * V, C, H, W)
                    features = self.model(images_reshaped)
                    features = features.view(B, V, -1)
                    # Concatenar vistas (no promediar) para preservar información
                    embeddings = features.flatten(start_dim=1)  # [B, V*embedding_dim]
                    
                    # Calcular pérdida y accuracy según el objetivo
                    if self.objective == 'classification':
                        outputs = self.head_layer(embeddings)
                        loss = criterion(outputs, labels)
                        _, predicted = outputs.max(1)
                        # Guardar embeddings sin proyección para recalls
                        all_embeddings.append(embeddings.cpu())
                        all_labels.append(labels.cpu())
                    elif self.objective == 'ArcFace':
                        projection_layer, arcface_layer = self.head_layer
                        embeddings = projection_layer(embeddings)
                        logits = arcface_layer(embeddings, labels)
                        loss = criterion(logits, labels)
                        # Para accuracy, usar ArcFaceInference 
                        arcface_inf = ArcFaceInference(
                            weight=arcface_layer.weight,
                            scale=arcface_layer.scale
                        )
                        inference_logits = arcface_inf(embeddings)
                        _, predicted = inference_logits.max(1)
                    elif self.objective == 'metric_learning':
                        # Proyectar embeddings a través de head_layer
                        projected_embeddings = self.head_layer(embeddings)
                            
                        if isinstance(criterion, TripletLoss):
                            # Calcular loss con hard mining 
                            batch_size = projected_embeddings.size(0)
                            dist_matrix = torch.cdist(projected_embeddings, projected_embeddings, p=2)
                            
                            anchors, positives, negatives = [], [], []
                            for i in range(batch_size):
                                pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
                                neg_mask = labels != labels[i]
                                
                                if pos_mask.any() and neg_mask.any():
                                    hardest_pos_idx = dist_matrix[i][pos_mask].argmax()
                                    pos_idx = torch.where(pos_mask)[0][hardest_pos_idx]
                                    hardest_neg_idx = dist_matrix[i][neg_mask].argmin()
                                    neg_idx = torch.where(neg_mask)[0][hardest_neg_idx]
                                    
                                    anchors.append(projected_embeddings[i])
                                    positives.append(projected_embeddings[pos_idx])
                                    negatives.append(projected_embeddings[neg_idx])
                            
                            if len(anchors) > 0:
                                loss = criterion(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))
                            else:
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            
                            # Placeholder para accuracy 
                            predicted = torch.zeros_like(labels)

                        elif isinstance(criterion, ContrastiveLoss):
                            # Calcular loss con pares 
                            batch_size = projected_embeddings.size(0)
                            pairs_1, pairs_2, pair_labels = [], [], []
                            
                            for i in range(batch_size):
                                for j in range(i + 1, batch_size):
                                    pairs_1.append(projected_embeddings[i])
                                    pairs_2.append(projected_embeddings[j])
                                    pair_labels.append(0 if labels[i] == labels[j] else 1)
                            
                            if len(pairs_1) > 0:
                                loss = criterion(
                                    torch.stack(pairs_1), 
                                    torch.stack(pairs_2), 
                                    torch.tensor(pair_labels, device=self.device, dtype=torch.float)
                                )
                            else:
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                            
                            # Placeholder para accuracy 
                            predicted = torch.zeros_like(labels)
                            
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

        if self.objective == 'metric_learning':
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
            use_similarity: Si usar predicción por similitud con centroides del train set. Si None, se decide automáticamente.

        Returns:
            Diccionario con métricas de evaluación.
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        if self.head_layer is not None:
            self.head_layer.eval()

        all_embeddings = []
        all_labels = []

        # Decidir método de predicción
        if use_similarity is None:
            use_similarity = (self.objective == 'metric_learning')

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                images = batch['images'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Extraer embeddings
                # Manejar múltiples vistas
                if images.ndim == 4:
                    images = images.unsqueeze(1)
                
                B, V, C, H, W = images.shape
                images_reshaped = images.view(B * V, C, H, W)
                features = self.model(images_reshaped)
                features = features.view(B, V, -1)
                # Concatenar vistas (no promediar) para preservar información
                embeddings = features.flatten(start_dim=1)  # [B, V*embedding_dim]
                
                # Si es metric_learning, proyectar embeddings
                if self.objective == 'metric_learning':
                    embeddings = self.head_layer(embeddings)
                elif self.objective == 'ArcFace':
                    projection_layer = self.head_layer[0]
                    embeddings = projection_layer(embeddings)
                    
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Realizar predicción según el método
            if use_similarity:
                # Metric learning: usar predicción por similitud con centroides
                predicted = self._predict_by_similarity(all_embeddings)
                accuracy = 100.0 * (predicted == all_labels).float().mean().item()
                
                return {
                    'accuracy': accuracy,
                    'total_samples': len(all_labels),
                    'predictions': predicted.numpy(),
                    'labels': all_labels.numpy(),
                    'method': 'similarity_centroids_from_train'
                }
            else:
                if self.objective == 'ArcFace':
                    # Usar ArcFaceInference para predicción sin labels
                    arcface_layer = self.head_layer[1]
                    arcface_inf = ArcFaceInference(
                        weight=arcface_layer.weight,
                        scale=arcface_layer.scale
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
        if images.ndim == 4:
            images = images.unsqueeze(1)

        B, V, C, H, W = images.shape  # batch, views, channels, height, width
        # Flatten views: [B*V, C, H, W]
        images_reshaped = images.view(B * V, C, H, W)
        
        self.model.eval()

        with torch.no_grad():
            # Extract features
            embeddings = self.model(images_reshaped)

        # Reshape back: [B, V, D]
        embeddings = embeddings.view(B, V, -1)
        # Average views: [B, D]
        embeddings = embeddings.mean(dim=1)
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        return embeddings
    
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