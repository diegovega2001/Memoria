"""
Módulo de modelo de visión para clasificación de vehículos multi-vista.

Este módulo proporciona una clase para manejar modelos de visión computacional
pre-entrenados, adaptándolos para clasificación de vehículos con múltiples vistas
e incluyendo funcionalidades de fine-tuning y extracción de embeddings.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constantes del modelo
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_WARMUP_EPOCHS = 0
WEIGHTS_FILENAME = 'vision_model.pth'
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
        classification_layer: Capa de clasificación final.
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
        weights: str = 'IMAGENET1K_V1',
        device: torch.device = None,
        dataset_dict: dict = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = True
    ) -> None:
        """
        Inicializa el modelo de visión multi-vista.

        Args:
            name: Nombre descriptivo del modelo.
            model_name: Nombre del modelo en torchvision.models.
            weights: Identificador de pesos pre-entrenados.
            device: Dispositivo de cómputo.
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
        self.classification_layer = self._create_classification_layer()

        # Mover a dispositivo
        self.to(device)

        logging.info(f"Inicializado {self.__class__.__name__}: {self.name}")
        logging.info(f"Modelo base: {self.model_name} con pesos {self.weights}")
        logging.info(f"Embedding dim: {self.embedding_dim}, Clases: {self.dataset.num_models}")

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
        """Reemplaza la capa final con Identity para extraer embeddings."""
        replaced = False
        
        # Intentar configuraciones conocidas
        for model_family, config in MODEL_CONFIGS.items():
            if model_family in self.model_name.lower():
                if hasattr(self.model, config['feature_attr']):
                    setattr(self.model, config['feature_attr'], nn.Identity())
                    replaced = True
                    break

        # Búsqueda general si no se encontró configuración específica
        if not replaced:
            for attr_name in SUPPORTED_MODEL_ATTRIBUTES:
                if hasattr(self.model, attr_name):
                    setattr(self.model, attr_name, nn.Identity())
                    replaced = True
                    break

        if not replaced:
            raise VisionModelError(
                f"No se pudo reemplazar la capa final para {self.model_name}"
            )

        logging.debug("Capa final reemplazada con Identity para extracción de embeddings")

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
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
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
            # Configuración de entrenamiento
            history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
            best_val_acc = 0.0
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
                val_loss, val_accuracy = self._validate_epoch(criterion, epoch, epochs)

                # Actualizar historial
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                # Log de progreso
                logging.info(
                    f'Epoch {epoch+1}/{epochs} | '
                    f'Train Loss: {train_loss:.4f} | '
                    f'Val Loss: {val_loss:.4f} | '
                    f'Val Acc: {val_accuracy:.2f}%'
                )

                # Guardar mejor modelo
                if save_best and val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    if checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, epoch, val_accuracy, 'best')

                # Early stopping
                if early_stopping:
                    if val_accuracy > best_val_acc - early_stop_min_delta:
                        patience_counter = 0
                        best_val_acc = max(best_val_acc, val_accuracy)
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stop_patience:
                        logging.info(f"Early stopping triggered at epoch {epoch+1}")
                        break

            logging.info(f"Fine-tuning completado. Mejor Val Acc: {best_val_acc:.2f}%")
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

    def _train_epoch(
        self,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int
    ) -> float:
        """Ejecuta una época de entrenamiento."""
        self.model.train()
        self.classification_layer.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)

        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', leave=False) as pbar:
            for batch in pbar:
                images = batch['images']
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                # Forward pass
                embeddings = torch.flatten(
                    self.model(images.to(self.device)), 
                    start_dim=1
                )
                
                outputs = self.classification_layer(embeddings)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def _validate_epoch(self, criterion: nn.Module, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """Ejecuta una época de validación."""
        self.model.eval()
        self.classification_layer.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            with tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]', leave=False) as pbar:
                for batch in pbar:
                    images = batch['images']
                    labels = batch['labels'].to(self.device)

                    # Forward pass
                    embeddings = torch.flatten(
                        self.model(images.to(self.device)), 
                        start_dim=1
                    )
                    
                    outputs = self.classification_layer(embeddings)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    
                    # Accuracy
                    _, predicted = outputs.max(1)
                    total_samples += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    current_acc = 100 * correct / total_samples
                    pbar.set_postfix({'acc': f'{current_acc:.2f}%'})

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total_samples
        
        return avg_loss, accuracy

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evalúa el modelo en un conjunto de datos.

        Args:
            dataloader: DataLoader a evaluar. Si None, usa val_loader.

        Returns:
            Diccionario con métricas de evaluación.
        """
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        self.classification_layer.eval()

        correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                images = batch['images']
                labels = batch['labels'].to(self.device)

                embeddings = torch.flatten(
                    self.model(images.to(self.device)), 
                    start_dim=1
                )
                
                outputs = self.classification_layer(embeddings)
                _, predicted = outputs.max(1)

                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculo del accuracy manejando 0 predicciones correctas
        eps = 1e-10
        accuracy = max(100 * correct / total_samples, eps) if total_samples > 0 else eps
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total_samples,
            'predictions': all_predictions,
            'labels': all_labels
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
        
        model_path = save_path / WEIGHTS_FILENAME

        # Preparar checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classification_layer_state_dict': self.classification_layer.state_dict(),
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
        model_path = load_path / WEIGHTS_FILENAME

        if not model_path.exists():
            raise VisionModelError(f"No se encontró modelo en: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Cargar estados
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.classification_layer.load_state_dict(checkpoint['classification_layer_state_dict'])
            
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
            'classification_layer_state_dict': self.classification_layer.state_dict(),
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
        dataset: Dataset con divisiones.
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