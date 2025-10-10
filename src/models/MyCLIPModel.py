"""
Módulo de modelo CLIP para clustering de vehículos multi-vista con descripción textual.

Este módulo proporciona una clase para manejar modelos CLIP (Contrastive Language-Image Pre-training)
adaptándolos para el clustering de vehículos con múltiples vistas e información textual,
incluyendo funcionalidades de fine-tuning por fases y extracción de embeddings multimodales.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup

from .Criterions import ContrastiveLoss

from src.defaults import (
    DEFAULT_BATCH_SIZE, 
    DEFAULT_NUM_WORKERS, 
    DEFAULT_PIN_MEMORY, 
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_USE_AMP,
    CLIP_CONFIGS
)


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CLIPModelError(Exception):
    """Excepción personalizada para errores del modelo CLIP."""
    pass


class MultiViewCLIPModel(nn.Module):
    """
    Modelo CLIP para clasificación/clustering de vehículos multi-vista con descripciones textuales.

    Esta clase adapta modelos CLIP pre-entrenados para trabajar con múltiples vistas de vehículos
    y descripciones textuales, proporcionando funcionalidades de fine-tuning por fases,
    extracción de embeddings multimodales y clasificación.

    El fine-tuning sigue una estrategia por fases inspirada en la investigación con CLIP:
    1. Text encoder: Ajusta la codificación del texto
    2. Projection layers: Ajusta las proyecciones imagen-texto
    3. Vision encoder (final layers): Ajusta capas finales de visión
    4. Projection layers (refinement): Refinamiento final de proyecciones

    Attributes:
        name: Nombre descriptivo del modelo.
        model_name: Nombre del modelo CLIP base.
        device: Dispositivo de cómputo (CPU/GPU).
        dataset: Dataset con las divisiones train/val/test.
        batch_size: Tamaño de batch para entrenamiento.
        model: Modelo CLIP de HuggingFace.
        processor: Procesador CLIP para imágenes y texto.
        embedding_dim: Dimensión de los embeddings CLIP.
        head_layer: Capa final para clasificación o metric learning.
        train_loader: DataLoader para entrenamiento.
        val_loader: DataLoader para validación.
        test_loader: DataLoader para prueba.

    Example:
        >>> model = MultiViewCLIPModel(
        ...     name="CLIP_MultiView",
        ...     model_name="clip-vit-base-patch32",
        ...     device=torch.device("cuda"),
        ...     dataset=car_dataset,
        ...     batch_size=32
        ... )
        >>> model.finetune_phase("text", criterion, optimizer, epochs=5)
        >>> model.finetune_phase("projection", criterion, optimizer, epochs=3)

    Raises:
        CLIPModelError: Para errores específicos del modelo.
        ValueError: Para parámetros inválidos.
    """
    def __init__(
        self,
        name: str,
        model_name: str = 'clip-vit-base-patch32',
        device: torch.device = None,
        objective: str = 'CLIP',
        dataset_dict: dict = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY
    ) -> None:
        """
        Inicializa el modelo CLIP multi-vista.

        Args:
            name: Nombre descriptivo del modelo.
            model_name: Nombre del modelo CLIP (key de CLIP_CONFIGS).
            device: Dispositivo de cómputo.
            objective: El objetivo del modelo (Solo CLIP).
            dataset_dict: Diccionario con el dataset y los dataloaders con divisiones train/val/test.
            batch_size: Tamaño de batch.
            num_workers: Número de workers para DataLoaders.
            pin_memory: Si usar pin_memory en DataLoaders.

        Raises:
            CLIPModelError: Si hay errores de inicialización.
            ValueError: Si los parámetros son inválidos.
        """
        super().__init__()
        
        # Validar parámetros
        self._validate_parameters(name, model_name, device, dataset_dict, batch_size)

        # Configuración básica
        self.name = name
        self.model_name = model_name
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.objective = objective
        self.dataset = dataset_dict['dataset']
        self.train_loader = dataset_dict['train_loader']
        self.val_loader = dataset_dict['val_loader']
        self.test_loader = dataset_dict['test_loader']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Inicialización del modelo CLIP
        self.model, self.processor = self._initialize_clip_model()
        self.contrastive_criterion = ContrastiveLoss(clip=True)

        # Mover a dispositivo
        self.model.to(self.device)
        
        logging.info(f"Inicializado {self.__class__.__name__}: {self.name}")
        logging.info(f"Modelo base: {self.model_name}")

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
            raise ValueError("name debe ser un string no vacío")

        if model_name not in CLIP_CONFIGS:
            raise ValueError(f"model_name debe ser uno de: {list(CLIP_CONFIGS.keys())}")

        if device is not None and not isinstance(device, torch.device):
            raise ValueError("device debe ser una instancia de torch.device o None")

        if batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")
        
        if dataset_dict is None:
            raise ValueError("dataset_dict no puede ser None")
            
        if dataset_dict.get('train_loader') is None or dataset_dict.get('val_loader') is None or dataset_dict.get('test_loader') is None:
            raise ValueError("dataset_dict debe contener 'train_loader', 'val_loader' y 'test_loader'")
            
        if not isinstance(dataset_dict['train_loader'], DataLoader):
            raise ValueError("train_loader debe ser una instancia de DataLoader")
        if not isinstance(dataset_dict['val_loader'], DataLoader):
            raise ValueError("val_loader debe ser una instancia de DataLoader")
        if not isinstance(dataset_dict['test_loader'], DataLoader):
            raise ValueError("test_loader debe ser una instancia de DataLoader")

    def _initialize_clip_model(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Inicializa el modelo CLIP y su procesador."""
        try:
            config = CLIP_CONFIGS[self.model_name]
            hf_model_name = config['model_name']
            
            # Usar safetensors para evitar vulnerabilidad de seguridad en torch.load
            model = CLIPModel.from_pretrained(hf_model_name, use_safetensors=True).to(self.device)
            processor = CLIPProcessor.from_pretrained(hf_model_name)
            
            logging.info(f"Modelo CLIP cargado: {hf_model_name}")
            return model, processor
            
        except Exception as e:
            raise CLIPModelError(f"Error inicializando modelo CLIP '{self.model_name}': {e}")

    def extract_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extrae embeddings de imágenes multi-vista.        
        Args:
            images: Tensor de dimensiones [batch_size, nun_vistas, C, H, W]
                    o [batch_size, C, H, W] si es una vista.
            
        Returns:
            Embeddings de imágenes normalizados con las mismas dimensiones que el original
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)

        B, V, C, H, W = images.shape  # batch, views, channels, height, width
        # Flatten views: [B*V, C, H, W]
        images_reshaped = images.view(B * V, C, H, W)
        # Extract features
        image_embeddings = self.model.get_image_features(pixel_values=images_reshaped)
        # Reshape back: [B, V, D]
        image_embeddings = image_embeddings.view(B, V, -1)
        # Average views: [B, D]
        image_embeddings = image_embeddings.mean(dim=1)
        # Normalize
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

        return image_embeddings
    
    def extract_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Extrae embeddings de descripciones textuales,
        
        Args:
            texts: Lista de descripciones de texto
            
        Returns:
            Embeddings de texto normalizados 
        """
        text_inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        text_embeddings = self.model.get_text_features(**text_inputs)
        # Normalize
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        return text_embeddings

    def extract_joint_embeddings(self, images: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extrae embeddings imagen-texto para cálculo de similitud.
        
        Args:
            images: Tensor [batch_size, num_vistas, C, H, W]
            texts: Lista de descripciones de texto
            
        Returns:
            Tupla de (image_embeddings, text_embeddings)
        """
        image_embeddings = self.extract_image_embeddings(images=images)
        text_embeddings = self.extract_text_embeddings(texts=texts)

        return image_embeddings, text_embeddings
    
    def extract_embeddings(self, dataloader: DataLoader) -> torch.Tensor:
        """Extrae embeddings combinados image-text del dataloader.
        
        Args:
            dataloader: DataLoader con pares image-text 
            
        Returns:
            Embbings combinados [N, 2*embedding_dim] (primera mitad images, segunda texts)
        """
        self.model.eval()
        
        all_image_embeddings = []
        all_text_embeddings = []
        
        with torch.no_grad():  
            pbar = tqdm(dataloader, desc="Extrayendo embeddings", leave=False)
            for batch in pbar:
                images = batch['images'].to(self.device)  
                texts = batch['text_description']  
                image_embeddings, text_embeddings = self.extract_joint_embeddings(images=images, texts=texts)
                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
        
        # Concatenate all batches
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        
        # Combine image and text embeddings side by side
        combined_embeddings = torch.cat([all_image_embeddings, all_text_embeddings], dim=1)

        return combined_embeddings
    
    def extract_val_embeddings(self) -> torch.Tensor:
        """Extrae embeddings combinados del dataloader de validación."""
        return self.extract_embeddings(
            self.val_loader
        )

    def extract_test_embeddings(self) -> torch.Tensor:
        """Extrae"""
        return self.extract_embeddings(
            self.test_loader
        )

    # -------------------------------------------------------------------------
    # Estrategia de congelamiento/descongelamiento por fases
    # -------------------------------------------------------------------------
    
    def freeze_all(self) -> None:
        """Congela todos los parámetros del modelo CLIP."""
        for param in self.model.parameters():
            param.requires_grad = False
        logging.debug("Todos los parámetros del modelo CLIP congelados")

    def unfreeze_text_encoder(self) -> None:
        """Descongela solo el text encoder."""
        self.freeze_all()
        for param in self.model.text_model.parameters():
            param.requires_grad = True
        logging.debug("Text encoder descongelado")

    def unfreeze_projection_layers(self) -> None:
        """Descongela las capas de proyección (text_projection y visual_projection)."""
        self.freeze_all()
        for param in self.model.text_projection.parameters():
            param.requires_grad = True
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True
        logging.debug("Projection layers descongeladas")

    def unfreeze_vision_final_layers(self, num_layers: int = 1) -> None:
        """
        Descongela las últimas capas del vision encoder.
        
        Args:
            num_layers: Número de capas finales a descongelar.
        """
        self.freeze_all()
        vision_layers = self.model.vision_model.encoder.layers
        for layer in vision_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        logging.debug(f"Últimas {num_layers} capas del vision encoder descongeladas")

    def get_trainable_params_count(self) -> Dict[str, int]:
        """
        Cuenta los parámetros entrenables actuales.
        
        Returns:
            Diccionario con conteo de parámetros.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }

    # -------------------------------------------------------------------------
    # Fine-tuning por fases
    # -------------------------------------------------------------------------

    def finetune(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int = DEFAULT_FINETUNE_EPOCHS,
        warmup_steps: int = 0,
        early_stopping: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        temperature: float = 0.07,
        use_amp: bool = DEFAULT_USE_AMP
    ) -> Dict[str, List[float]]:
        total_steps = epochs * len(self.train_loader)
        """
        Método de finetuning estandar para todas las fases/modelos.
        Returns:
            Diccionario con historial de entrenamiento.
        """
        try: 
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_recall@1': [],
                'val_recall@5': []
            }

            best_val_metric = 0.0
            patience_counter = 0

            warmup_scheduler = None
            if warmup_steps > 0:
                warmup_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
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
            
            logging.info(f"Iniciando fine-tuning: {epochs} épocas, warmup: {warmup_steps}, AMP: {use_amp}")

            for epoch in range(epochs):
                train_loss = self._train_epoch(optimizer, epoch, epochs, temperature, use_amp=use_amp, scaler=scaler)
                
                if warmup_scheduler:
                    warmup_scheduler.step()
                
                val_loss, val_recalls = self._validate_epoch(epoch, epochs, temperature)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_recall@1'].append(val_recalls[1])
                history['val_recall@5'].append(val_recalls[5])

                current_metric = val_recalls[1]

                log_msg = (
                        f'Epoch {epoch+1}/{epochs} | '
                        f'Train Loss: {train_loss:.4f} | '
                        f'Val Loss: {val_loss:.4f} |'
                        f'Recall@1: {val_recalls[1]:.2f}% | Recall@5: {val_recalls[5]:.2f}%'
                    )
                
                logging.info(log_msg)

                if save_best and current_metric > best_val_metric:
                    best_val_metric = current_metric
                    if checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir, epoch, current_metric, 'best')
                
                if early_stopping:
                    if current_metric > best_val_metric - early_stop_min_delta:
                        patience_counter = 0
                        best_val_metric = current_metric
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stop_patience:
                        logging.info(f'Early stopping triggered at epoch {epoch+1} based on Recall@1')
                        break

            logging.info(f'Fine-Tuning completado.')
            return history
    
        except Exception as e:
            raise CLIPModelError(f'Error durante fine-tuning: {e}') from e
        
    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
        temperature: float = 0.07,
        use_amp: bool = False,
        scaler: Optional[torch.amp.GradScaler] = None
    ) -> float:
        """Ejecuta una época de entrenamiento con soporte opcional para AMP."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", leave=False)
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            texts = batch['text_description']

            optimizer.zero_grad()
        
            # Forward pass con AMP si está habilitado
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                image_embeddings, text_embeddings = self.extract_joint_embeddings(images, texts)

                # Compute similarity and loss
                logits_per_image = (image_embeddings @ text_embeddings.t()) / temperature
                logits_per_text = logits_per_image.t()

                loss = self.contrastive_criterion(logits_per_image, logits_per_text, None)

            # Backward pass con GradScaler si AMP está habilitado
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches

    def _validate_epoch(
        self, 
        epoch: int, 
        total_epochs: int,
        temperature: float
    ) -> Tuple[float, Dict[int, float]]:
        """Ejecuta una época de validación."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_image_embeddings = []
        all_text_embeddings = []
        all_labels = []
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['images'].to(self.device)
                texts = batch['text_description']
                labels = batch['labels'].to(self.device)

                # Extract embeddings (no gradients needed)
                image_embeddings, text_embeddings = self.extract_joint_embeddings(images=images, texts=texts)

                # Compute similarity and loss
                logits_per_image = (image_embeddings @ text_embeddings.t()) / temperature
                logits_per_text = logits_per_image.t()

                loss = self.contrastive_criterion(logits_per_image, logits_per_text, None)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
                all_labels.append(labels.cpu())
                
                pbar.set_postfix({'loss': loss.item()})
        
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        recalls = self._calculate_recall_metrics(
            all_image_embeddings, 
            all_text_embeddings, 
            all_labels
        )
        
        avg_loss = total_loss / num_batches
    
        return avg_loss, recalls
    
    def finetune_phase(
        self,
        phase: str,
        optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        warmup_steps: int = 200,
        early_stopping: Optional[Dict[str, Any]] = None,
        save_best: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_amp: bool = DEFAULT_USE_AMP,
        num_vision_layers: int = 1
    ) -> Dict[str, List[float]]:
        """
        Ejecuta fine-tuning en una fase específica.
        
        Args:
            phase: Fase de fine-tuning ('text', 'projection', 'vision', 'projection_refine').
            criterion: Función de pérdida.
            optimizer: Optimizador.
            epochs: Número de épocas para esta fase.
            scheduler: Learning rate scheduler.
            early_stopping: Configuración de early stopping.
            save_best: Si guardar el mejor modelo.
            checkpoint_dir: Directorio para checkpoints.
            
        Returns:
            Diccionario con historial de entrenamiento.
        """
        # Configurar congelamiento según la fase
        if phase == 'text':
            self.unfreeze_text_encoder()
        elif phase == 'projection':
            self.unfreeze_projection_layers()
        elif phase == 'vision':
            self.unfreeze_vision_final_layers(num_layers=num_vision_layers)
        elif phase == 'projection_refine':
            self.unfreeze_projection_layers()
        else:
            raise ValueError(f"Fase '{phase}' no soportada. Use 'text', 'projection', 'vision' o 'projection_refine'.")
        
        # Mostrar parámetros entrenables
        params_info = self.get_trainable_params_count()
        logging.info(f"Fase '{phase}': {params_info['trainable']:,} parámetros entrenables de {params_info['total']:,} totales")
        
        # Ejecutar entrenamiento
        history = self.finetune(
            optimizer=optimizer,
            epochs=epochs,
            warmup_steps=warmup_steps,
            early_stopping=early_stopping,
            save_best=save_best,
            checkpoint_dir=checkpoint_dir,
            use_amp=use_amp
        )
        
        return history
    
    def _calculate_recall_metrics(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """
        Calcula métricas Recall@K para image-to-text retrieval.
        
        Args:
            image_embeddings: [N, embed_dim]
            text_embeddings: [N, embed_dim]
            labels: [N] - class labels
            k_values: Lista de k para Recall@K
        
        Returns:
            Dict con recall@k para cada k
        """
        # Compute similarity matrix: [N, N]
        similarity = image_embeddings @ text_embeddings.t()
        
        recalls = {}
        
        for k in k_values:
            correct = 0
            for i in range(len(labels)):
                # Get top-k most similar texts for this image
                top_k_indices = similarity[i].topk(k).indices
                
                # Check if any of top-k matches the label
                if labels[i] in labels[top_k_indices]:
                    correct += 1
            
            recalls[k] = 100.0 * correct / len(labels)
        
        return recalls

    def evaluate(
        self, 
        dataloader: Optional[DataLoader] = None,
        temperature: float = 0.07
    ) -> Dict[str, float]:
        """
        Evalúa el modelo en un dataloader usando métricas de retrieval.
        
        Args:
            dataloader: DataLoader para evaluación (por defecto test_loader).
            temperature: Temperature para scaling de logits.
            
        Returns:
            Diccionario con métricas de evaluación (loss, recall@1, recall@5, recall@10).
        """
        if dataloader is None:
            dataloader = self.test_loader
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        all_image_embeddings = []
        all_text_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluando", leave=False):
                images = batch['images'].to(self.device)
                texts = batch['text_description']
                labels = batch['labels'].to(self.device)

                # Extract embeddings
                image_embeddings, text_embeddings = self.extract_joint_embeddings(images=images, texts=texts)

                # Compute loss
                logits_per_image = (image_embeddings @ text_embeddings.t()) / temperature
                logits_per_text = logits_per_image.t()
                loss = self.contrastive_criterion(logits_per_image, logits_per_text, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_image_embeddings.append(image_embeddings.cpu())
                all_text_embeddings.append(text_embeddings.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all embeddings
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate recall metrics
        recalls = self._calculate_recall_metrics(
            all_image_embeddings,
            all_text_embeddings,
            all_labels
        )
        
        avg_loss = total_loss / num_batches
        
        results = {
            'loss': avg_loss,
            'recall@1': recalls.get(1, 0.0),
            'recall@5': recalls.get(5, 0.0),
        }
        
        logging.info(
            f"Evaluación: Loss = {avg_loss:.4f} | "
            f"Recall@1 = {results['recall@1']:.2f}% | "
            f"Recall@5 = {results['recall@5']:.2f}% | "
        )
        return results

    def save_model(self, save_path: Union[str, Path], **kwargs) -> None:
        """Guarda el modelo."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, save_path)
        logging.info(f"Modelo guardado en: {save_path}")

    def load_model(self, load_path: Union[str, Path]) -> Dict[str, Any]:
        """Carga el modelo."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logging.info(f"Modelo cargado desde: {load_path}")
        return checkpoint

    def _save_checkpoint(
        self, 
        checkpoint_dir: Union[str, Path], 
        epoch: int, 
        accuracy: float, 
        suffix: str = ''
    ) -> None:
        """Guarda un checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        filename = f"checkpoint_epoch{epoch}_{suffix}.pth" if suffix else f"checkpoint_epoch{epoch}.pth"
        self.save_model(checkpoint_dir / filename, epoch=epoch, accuracy=accuracy)

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo."""
        params_info = self.get_trainable_params_count()
        
        return {
            'name': self.name,
            'model_name': self.model_name,
            'num_classes': self.dataset.num_models,
            'device': str(self.device),
            'total_parameters': params_info['total'],
            'trainable_parameters': params_info['trainable'],
            'frozen_parameters': params_info['frozen']
        }

    def __str__(self) -> str:
        """Representación string del modelo."""
        info = self.get_model_info()

        return (
            f"{self.__class__.__name__}(\n"
            f"  name='{info['name']}',\n"
            f"  model='{info['model_name']}',\n"
            f"  num_classes={info['num_classes']},\n"
            f"  trainable_params={info['trainable_parameters']:,}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Representación detallada del modelo."""
        return self.__str__()


# Funciones de conveniencia
def create_clip_model(
    model_name: str,
    dataset_dict: dict,
    device: Optional[torch.device] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **kwargs
) -> MultiViewCLIPModel:
    """
    Función de conveniencia para crear un modelo CLIP.

    Args:
        model_name: Nombre del modelo CLIP (key de CLIP_CONFIGS).
        dataset_dict: Diccionario con dataset y dataloaders.
        device: Dispositivo de cómputo.
        batch_size: Tamaño de batch.
        **kwargs: Argumentos adicionales.

    Returns:
        Modelo CLIP inicializado.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    name = kwargs.pop('name', f"CLIP_{model_name.upper()}_MultiView")

    return MultiViewCLIPModel(
        name=name,
        model_name=model_name,
        device=device,
        dataset_dict=dataset_dict,
        batch_size=batch_size,
        **kwargs
    )
