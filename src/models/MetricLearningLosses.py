"""
Módulo de pérdidas para Metric Learning usando pytorch-metric-learning.
"""

from __future__ import annotations

import logging
import warnings
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners, distances, reducers

from src.defaults import (
    DEFAULT_TRIPLET_MARGIN, 
    DEFAULT_CONTRASTIVE_MARGIN, 
    DEFAULT_ARCFACE_SCALE, 
    DEFAULT_ARCFACE_MARGIN
)

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TripletLossWrapper(nn.Module):
    """
    Wrapper para TripletMarginLoss de pytorch-metric-learning.
    
    Utiliza la implementación optimizada de la librería que incluye:
    - Múltiples estrategias de selección de triplets
    - Mining automático de triplets difíciles
    - Mejor estabilidad numérica
    
    Args:
        margin: Margen mínimo entre distancias positivas y negativas.
        distance: Métrica de distancia ('euclidean', 'cosine', 'dot').
        reducer: Cómo reducir las pérdidas ('mean', 'sum').
    
    Example:
        >>> loss_fn = TripletLossWrapper(margin=0.2)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(
        self, 
        margin: float = DEFAULT_TRIPLET_MARGIN,
        distance: str = 'euclidean',
        reducer: str = 'mean'
        ):
        super(TripletLossWrapper, self).__init__()
        
        # Configurar distancia
        if distance == 'euclidean':
            distance_obj = distances.LpDistance(p=2, normalize_embeddings=True)
        elif distance == 'cosine':
            distance_obj = distances.CosineSimilarity()
        elif distance == 'dot':
            distance_obj = distances.DotProductSimilarity()
        else:
            raise ValueError(f"Distancia '{distance}' no soportada")
        
        # Crear loss de pytorch-metric-learning
        self.loss_fn = losses.TripletMarginLoss(
            margin=margin,
            distance=distance_obj,
            reducer=self._get_reducer(reducer),
        )
        
        self.margin = margin
        self.distance_type = distance
        
        logging.info(f"TripletLossWrapper inicializado: margin={margin}, distance={distance}")
    
    def _get_reducer(self, reducer: str):
        """Obtiene el reducer de pytorch-metric-learning."""
        if reducer == 'mean':
            return reducers.MeanReducer()
        elif reducer == 'sum':
            return reducers.SumReducer()
        else:
            return reducers.MeanReducer()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            labels: Etiquetas de clase [batch_size]
            
        Returns:
            Pérdida triplet calculada.
        """
        return self.loss_fn(embeddings, labels)


class ContrastiveLossWrapper(nn.Module):
    """
    Wrapper para ContrastiveLoss de pytorch-metric-learning.
    
    Implementa contrastive loss que:
    - Minimiza distancia entre pares de la misma clase
    - Maximiza distancia entre pares de clases diferentes
    
    Args:
        pos_margin: Margen para pares positivos.
        neg_margin: Margen para pares negativos.
        distance: Métrica de distancia ('euclidean', 'cosine').
    
    Example:
        >>> loss_fn = ContrastiveLossWrapper(pos_margin=0, neg_margin=1.0)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(
        self,
        pos_margin: float = 0,
        neg_margin: float = DEFAULT_CONTRASTIVE_MARGIN,
        distance: str = 'euclidean'
    ):
        super(ContrastiveLossWrapper, self).__init__()
        
        # Configurar distancia
        if distance == 'euclidean':
            distance_obj = distances.LpDistance(p=2, normalize_embeddings=True)
        elif distance == 'cosine':
            distance_obj = distances.CosineSimilarity()
        else:
            raise ValueError(f"Distancia '{distance}' no soportada")
        
        # Crear loss de pytorch-metric-learning
        self.loss_fn = losses.ContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            distance=distance_obj
        )
        
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
        logging.info(f"ContrastiveLossWrapper inicializado: neg_margin={neg_margin}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            labels: Etiquetas de clase [batch_size]
            
        Returns:
            Pérdida contrastiva calculada.
        """
        return self.loss_fn(embeddings, labels)


class ArcFaceLossWrapper(nn.Module):
    """
    Wrapper para ArcFaceLoss de pytorch-metric-learning.
    
    ArcFace agrega un margen angular en el espacio de características
    para mejorar la discriminación entre clases.
    
    Args:
        num_classes: Número de clases.
        embedding_dim: Dimensión de los embeddings.
        margin: Margen angular en grados.
        scale: Factor de escala para los logits.
    
    Example:
        >>> loss_fn = ArcFaceLossWrapper(num_classes=100, embedding_dim=512)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = DEFAULT_ARCFACE_MARGIN,
        scale: float = DEFAULT_ARCFACE_SCALE
    ):
        super(ArcFaceLossWrapper, self).__init__()
        
        # Convertir margen de radianes a grados si es necesario
        margin_degrees = math.degrees(margin) if margin < 1 else margin
        
        # Crear ArcFaceLoss de pytorch-metric-learning
        self.loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=margin_degrees,
            scale=scale
        )
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin_degrees
        self.scale = scale
        
        logging.info(f"ArcFaceLossWrapper inicializado: classes={num_classes}, "
                    f"embedding_dim={embedding_dim}, margin={margin_degrees}°, scale={scale}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            labels: Etiquetas de clase [batch_size]
            
        Returns:
            Pérdida ArcFace calculada.
        """
        return self.loss_fn(embeddings, labels)
    
    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Obtiene logits para inferencia (sin labels).
        
        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Normalizar embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Normalizar pesos de la capa final (W es [num_classes, embedding_dim])
        # Transponer para F.linear: necesita [embedding_dim, num_classes]
        weight = F.normalize(self.loss_fn.W, p=2, dim=1)
        weight_t = weight.t()  # [embedding_dim, num_classes]
        
        # Calcular cosine similarity: [batch, emb_dim] @ [emb_dim, num_classes] = [batch, num_classes]
        logits = F.linear(embeddings_norm, weight_t) * self.scale
        
        return logits


class NTXentLossWrapper(nn.Module):
    """
    Wrapper para NTXentLoss (Normalized Temperature-scaled Cross Entropy Loss).
    
    También conocido como InfoNCE loss, usado en SimCLR y CLIP.
    Ideal para contrastive learning con grandes batches.
    
    Args:
        temperature: Parámetro de temperatura para suavizar distribución.
    
    Example:
        >>> loss_fn = NTXentLossWrapper(temperature=0.07)
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(self, temperature: float = 0.07):
        super(NTXentLossWrapper, self).__init__()
        
        self.loss_fn = losses.NTXentLoss(temperature=temperature)
        self.temperature = temperature
        
        logging.info(f"NTXentLossWrapper inicializado: temperature={temperature}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Embeddings [batch_size, embedding_dim]
            labels: Etiquetas de clase [batch_size]
            
        Returns:
            Pérdida NTXent calculada.
        """
        return self.loss_fn(embeddings, labels)


class MultiSimilarityLossWrapper(nn.Module):
    """
    Wrapper para MultiSimilarityLoss.
    
    Loss avanzado que considera múltiples pares de similitud
    simultáneamente para mejor convergencia.
    
    Args:
        alpha: Peso para pares positivos.
        beta: Peso para pares negativos.
        base: Base para la similitud.
    
    Example:
        >>> loss_fn = MultiSimilarityLossWrapper()
        >>> loss = loss_fn(embeddings, labels)
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 50.0, base: float = 0.5):
        super(MultiSimilarityLossWrapper, self).__init__()
        
        self.loss_fn = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        
        logging.info(f"MultiSimilarityLossWrapper inicializado: alpha={alpha}, beta={beta}")
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.loss_fn(embeddings, labels)


class CLIPLoss(nn.Module):
    """
    Implementación de CLIP's contrastive loss (imagen-texto).
    
    Calcula cross-entropy bidireccional entre logits de imagen y texto.
    Cada imagen debe coincidir con su texto correspondiente en el batch.
    
    Args:
        reduction: Tipo de reducción ('mean', 'sum', 'none').
    
    Example:
        >>> loss_fn = CLIPLoss()
        >>> loss = loss_fn(logits_per_image, logits_per_text)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(CLIPLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass para CLIP loss.
        
        Args:
            logits_per_image: Logits imagen→texto [batch_size, batch_size]
            logits_per_text: Logits texto→imagen [batch_size, batch_size]
            
        Returns:
            Pérdida CLIP calculada.
        """
        batch_size = logits_per_image.shape[0]
        
        # Ground truth: cada imagen coincide con su texto en la misma posición
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)
        
        # Cross-entropy en ambas direcciones
        loss_i2t = F.cross_entropy(logits_per_image, ground_truth, reduction=self.reduction)
        loss_t2i = F.cross_entropy(logits_per_text, ground_truth, reduction=self.reduction)
        
        # Promediar ambas direcciones
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss


def create_metric_learning_loss(
    loss_type: str,
    num_classes: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function para crear pérdidas de metric learning.
    
    Args:
        loss_type: Tipo de pérdida ('triplet', 'contrastive', 'arcface', 'ntxent', 
                   'multisimilarity', 'clip').
        num_classes: Número de clases (requerido para 'arcface').
        embedding_dim: Dimensión de embeddings (requerido para 'arcface').
        **kwargs: Argumentos adicionales específicos de cada pérdida.
        
    Returns:
        Módulo de pérdida configurado.
        
    Raises:
        ValueError: Si loss_type no es soportado o faltan parámetros requeridos.
        
    Example:
        >>> # Triplet Loss
        >>> loss = create_metric_learning_loss('triplet', margin=0.2)
        >>> 
        >>> # ArcFace Loss
        >>> loss = create_metric_learning_loss('arcface', num_classes=100, embedding_dim=512)
        >>> 
        >>> # CLIP Loss
        >>> loss = create_metric_learning_loss('clip')
    """
    loss_type_lower = loss_type.lower()
    
    if loss_type_lower in ['triplet', 'tripletloss']:
        margin = kwargs.get('margin', DEFAULT_TRIPLET_MARGIN)
        distance = kwargs.get('distance', 'euclidean')
        return TripletLossWrapper(margin=margin, distance=distance)
    
    elif loss_type_lower in ['contrastive', 'contrastiveloss']:
        neg_margin = kwargs.get('margin', DEFAULT_CONTRASTIVE_MARGIN)
        pos_margin = kwargs.get('pos_margin', 0)
        distance = kwargs.get('distance', 'euclidean')
        return ContrastiveLossWrapper(
            pos_margin=pos_margin, 
            neg_margin=neg_margin, 
            distance=distance
        )
    
    elif loss_type_lower in ['arcface', 'arcfaceloss']:
        if num_classes is None or embedding_dim is None:
            raise ValueError("num_classes y embedding_dim son requeridos para ArcFace")
        margin = kwargs.get('margin', DEFAULT_ARCFACE_MARGIN)
        scale = kwargs.get('scale', DEFAULT_ARCFACE_SCALE)
        return ArcFaceLossWrapper(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=margin,
            scale=scale
        )
    
    elif loss_type_lower in ['ntxent', 'ntxentloss', 'infonce']:
        temperature = kwargs.get('temperature', 0.07)
        return NTXentLossWrapper(temperature=temperature)
    
    elif loss_type_lower in ['multisimilarity', 'multisimilarityloss']:
        alpha = kwargs.get('alpha', 2.0)
        beta = kwargs.get('beta', 50.0)
        base = kwargs.get('base', 0.5)
        return MultiSimilarityLossWrapper(alpha=alpha, beta=beta, base=base)
    
    elif loss_type_lower in ['clip', 'cliploss']:
        return CLIPLoss()
    
    else:
        raise ValueError(
            f"Tipo de pérdida '{loss_type}' no soportado. "
            f"Opciones: 'triplet', 'contrastive', 'arcface', 'ntxent', 'multisimilarity', 'clip'"
        )


def create_miner(miner_type: str = 'all', **kwargs):
    """
    Factory function para crear miners de pytorch-metric-learning.
    
    Los miners seleccionan automáticamente pares/triplets difíciles
    durante el entrenamiento para mejorar convergencia.
    
    Args:
        miner_type: Tipo de miner ('all', 'hard', 'semihard', 'distance', 'multisimilarity').
        **kwargs: Argumentos específicos del miner.
        
    Returns:
        Miner configurado o None.
        
    Example:
        >>> miner = create_miner('semihard', margin=0.2)
        >>> hard_pairs = miner(embeddings, labels)
    """
    miner_type_lower = miner_type.lower()
    
    if miner_type_lower == 'all':
        # No mining, usar todos los pares/triplets
        return None
    
    elif miner_type_lower == 'hard':
        # Hard mining: pares/triplets más difíciles
        return miners.BatchHardMiner()
    
    elif miner_type_lower == 'semihard':
        # Semi-hard mining para triplets
        margin = kwargs.get('margin', DEFAULT_TRIPLET_MARGIN)
        return miners.TripletMarginMiner(margin=margin, type_of_triplets='semihard')
    
    elif miner_type_lower == 'distance':
        # Distance-based mining
        cutoff = kwargs.get('cutoff', 0.5)
        return miners.DistanceWeightedMiner(cutoff=cutoff)
    
    elif miner_type_lower == 'multisimilarity':
        # Multi-similarity mining
        epsilon = kwargs.get('epsilon', 0.1)
        return miners.MultiSimilarityMiner(epsilon=epsilon)
    
    else:
        logging.warning(f"Miner type '{miner_type}' no reconocido. Usando None (all pairs).")
        return None
