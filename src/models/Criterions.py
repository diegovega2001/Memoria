"""
Módulo de modelo de visión para clasificación de vehículos multi-vista.

Este módulo proporciona una clase para manejar modelos de visión computacional
pre-entrenados, adaptándolos para clasificación de vehículos con múltiples vistas
e incluyendo funcionalidades de fine-tuning y extracción de embeddings.
"""

from __future__ import annotations

import logging
import warnings
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.defaults import DEFAULT_TRIPLET_MARGIN, DEFAULT_CONTRASTIVE_MARGIN, DEFAULT_ARCFACE_SCALE, DEFAULT_ARCFACE_MARGIN


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TripletLoss(nn.Module):
    """
    Implementación de Triplet Loss para embeddings.
    
    La función de pérdida Triplet Loss toma tres embeddings: un ancla (anchor), 
    un positivo (de la misma clase que el ancla) y un negativo (de una clase diferente).
    Busca minimizar la distancia entre ancla-positivo y maximizar la distancia 
    ancla-negativo por un margen específico.
    
    Loss = max(0, d(a,p) - d(a,n) + margin)
    
    Args:
        margin: Margen mínimo entre distancias positivas y negativas.
        p: Norma para el cálculo de distancia (default: 2 para distancia euclidiana).
        reduction: Tipo de reducción ('mean', 'sum', 'none').
    
    Example:
        >>> triplet_loss = TripletLoss(margin=1.0)
        >>> loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
    """
    
    def __init__(self, margin: float = DEFAULT_TRIPLET_MARGIN, p: float = 2.0, reduction: str = 'mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.reduction = reduction
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de Triplet Loss.
        
        Args:
            anchor: Embeddings ancla [batch_size, embedding_dim]
            positive: Embeddings positivos [batch_size, embedding_dim]  
            negative: Embeddings negativos [batch_size, embedding_dim]
            
        Returns:
            Pérdida triplet calculada.
        """
        # Normalizar embeddings para mejor estabilidad
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Calcular distancias
        pos_dist = F.pairwise_distance(anchor, positive, p=self.p)
        neg_dist = F.pairwise_distance(anchor, negative, p=self.p)
        
        # Aplicar margen
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Aplicar reducción
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Implementación de Contrastive Loss para embeddings.
    
    Soporta dos modos:
    1. CLIP mode (clip=True): Trabaja con matrices de similitud (logits) usando CrossEntropyLoss
       - Cada imagen debe coincidir con su texto correspondiente en el batch (diagonal)
       - Ground truth son los índices del batch (torch.arange)
       
    2. Traditional mode (clip=False): Trabaja con pares de embeddings
       - Minimiza distancia entre pares positivos (misma clase)
       - Maximiza distancia entre pares negativos (diferentes clases) hasta un margen
    
    Loss tradicional = (1-Y) * 1/2 * (D^2) + (Y) * 1/2 * max(0, margin-D)^2
    donde Y=0 para pares positivos, Y=1 para pares negativos
    
    Loss CLIP = (CrossEntropy(logits_i2t, targets) + CrossEntropy(logits_t2i, targets)) / 2
    donde targets = torch.arange(batch_size)
    
    Args:
        margin: Margen mínimo para pares negativos (solo para modo tradicional).
        reduction: Tipo de reducción ('mean', 'sum', 'none').
        clip: Si usar modo CLIP (logits) o modo tradicional (embeddings).
    
    Example:
        >>> # Modo tradicional
        >>> contrastive_loss = ContrastiveLoss(margin=1.0, clip=False)
        >>> loss = contrastive_loss(embeddings1, embeddings2, labels)
        >>> 
        >>> # Modo CLIP
        >>> contrastive_loss = ContrastiveLoss(clip=True)
        >>> loss = contrastive_loss(logits_per_image, logits_per_text, None)
    """
    
    def __init__(
        self, 
        margin: float = DEFAULT_CONTRASTIVE_MARGIN, 
        reduction: str = 'mean',
        clip: bool = True
    ):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.clip = clip
    
    def forward(
        self, 
        input1: torch.Tensor, 
        input2: torch.Tensor, 
        label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass de Contrastive Loss.
        
        Args:
            input1: Si clip=True: logits_per_image [batch_size, batch_size]
                   Si clip=False: embeddings1 [batch_size, embedding_dim]
            input2: Si clip=True: logits_per_text [batch_size, batch_size]
                   Si clip=False: embeddings2 [batch_size, embedding_dim]
            label: Si clip=True: None (usa torch.arange internamente)
                  Si clip=False: Etiquetas (0=misma clase, 1=diferentes clases) [batch_size]
            
        Returns:
            Pérdida contrastiva calculada.
        """
        if self.clip:
            # Modo CLIP: input1=logits_per_image, input2=logits_per_text
            return self._clip_contrastive_loss(input1, input2)
        else:
            # Modo tradicional: input1=embedding1, input2=embedding2
            if label is None:
                raise ValueError("label es requerido para modo tradicional (clip=False)")
            return self._traditional_contrastive_loss(input1, input2, label)
    
    def _clip_contrastive_loss(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula CLIP's contrastive loss.
        
        En CLIP, cada imagen debe coincidir con su texto correspondiente en el batch.
        La diagonal de la matriz de similitud contiene los pares correctos.
        
        Args:
            logits_per_image: Logits de imagen a texto [batch_size, batch_size]
            logits_per_text: Logits de texto a imagen [batch_size, batch_size]
            
        Returns:
            Pérdida contrastiva promediada en ambas direcciones.
        """
        batch_size = logits_per_image.shape[0]
        
        # Ground truth: cada imagen coincide con su texto en la misma posición del batch
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)
        
        # CrossEntropy en ambas direcciones
        loss_i2t = F.cross_entropy(logits_per_image, ground_truth, reduction=self.reduction)
        loss_t2i = F.cross_entropy(logits_per_text, ground_truth, reduction=self.reduction)
        
        # Promediar ambas direcciones
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss
    
    def _traditional_contrastive_loss(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula contrastive loss tradicional basado en distancia entre embeddings.
        
        Args:
            embedding1: Primer conjunto de embeddings [batch_size, embedding_dim]
            embedding2: Segundo conjunto de embeddings [batch_size, embedding_dim]
            label: Etiquetas (0=misma clase, 1=diferentes clases) [batch_size]
            
        Returns:
            Pérdida contrastiva calculada.
        """
        # Normalizar embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)
        
        # Calcular distancia euclidiana
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Aplicar pérdida contrastiva
        pos_loss = (1 - label) * torch.pow(distance, 2)
        neg_loss = label * torch.pow(F.relu(self.margin - distance), 2)
        
        loss = 0.5 * (pos_loss + neg_loss)
        
        # Aplicar reducción
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ArcFaceLoss(nn.Module):
    """
    Implementación de ArcFace Loss para embeddings de alta calidad.
    
    ArcFace mejora la pérdida Softmax tradicional agregando un margen angular
    en el espacio de características, lo que resulta en embeddings más discriminatorios
    y compactos, ideales para clustering y reconocimiento.
    
    La pérdida modifica el coseno del ángulo entre embedding y peso de clase:
    cos(θ + m) donde θ es el ángulo original y m es el margen angular.
    
    Args:
        embedding_dim: Dimensión de los embeddings de entrada.
        num_classes: Número de clases en el dataset.
        scale: Factor de escala para amplificar los logits.
        margin: Margen angular en radianes.
        easy_margin: Si usar margen fácil (evita problemas numéricos).
    
    Example:
        >>> arcface_loss = ArcFaceLoss(embedding_dim=512, num_classes=163)
        >>> logits = arcface_loss(embeddings, labels)
        >>> loss = F.cross_entropy(logits, labels)
    """
    
    def __init__(
        self, 
        embedding_dim: int, 
        num_classes: int, 
        scale: float = DEFAULT_ARCFACE_SCALE,
        margin: float = DEFAULT_ARCFACE_MARGIN, 
        easy_margin: bool = False
    ):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Capa lineal para generar logits (pesos son los centroides de clase)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-calcular valores para eficiencia
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de ArcFace Loss.
        
        Args:
            embeddings: Embeddings normalizados [batch_size, embedding_dim]
            labels: Etiquetas de clase [batch_size]
            
        Returns:
            Logits con margen angular aplicado [batch_size, num_classes]
        """
        # Normalizar embeddings y pesos
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calcular coseno del ángulo entre embedding y pesos de clase
        cosine = F.linear(embeddings, weight)
        
        # Calcular seno y aplicar margen angular
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Crear one-hot encoding para las etiquetas
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Aplicar margen solo a la clase verdadera
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale
        
        return logits

class ArcFaceInference(nn.Module):
    """
    Capa de inferencia para ArcFace que solo usa cosine similarity sin margen angular.
    Útil para evaluación donde no se tienen labels durante predicción.
    
    Args:
        weight: Matriz de pesos W de ArcFace entrenado (embedding_dim x num_classes).
        scale: Factor de escala (mismo usado durante entrenamiento).
    
    Example:
        >>> # Durante entrenamiento se usa ArcFaceLoss
        >>> arcface = ArcFaceLoss(embedding_dim=512, num_classes=163)
        >>> # Durante inferencia se usa ArcFaceInference con los pesos entrenados
        >>> arcface_inf = ArcFaceInference(arcface.weight, arcface.scale)
        >>> logits = arcface_inf(embeddings)  # No requiere labels
    """
    def __init__(self, weight: nn.Parameter, scale: float):
        super().__init__()
        self.weight = weight  # Compartir referencia a los pesos entrenados
        self.scale = scale
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Inferencia usando solo cosine similarity (sin margen angular).
        
        Args:
            embeddings: Embeddings normalizados (batch_size, embedding_dim).
            
        Returns:
            Logits escalados (batch_size, num_classes).
        """
        # Normalizar embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Normalizar pesos
        weight_norm = F.normalize(self.weight, p=2, dim=0)
        
        # Cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)
        
        # Escalar
        logits = cosine * self.scale
        
        return logits
    

def create_metric_learning_criterion(
    loss_type: str,
    embedding_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    **kwargs
) -> Union[TripletLoss, ContrastiveLoss, ArcFaceLoss, nn.CrossEntropyLoss]:
    """
    Función de conveniencia para crear funciones de pérdida para metric learning.
    
    Args:
        loss_type: Tipo de pérdida ('triplet', 'contrastive', 'arcface').
        embedding_dim: Dimensión de embeddings (requerido para ArcFace).
        num_classes: Número de clases (requerido para ArcFace).
        **kwargs: Argumentos adicionales específicos de cada pérdida.
        
    Returns:
        Función de pérdida configurada.
        
    Example:
        >>> # Para Triplet Loss
        >>> triplet_criterion = create_metric_learning_criterion('triplet', margin=1.0)
        >>> 
        >>> # Para ArcFace (retorna ArcFace layer + CrossEntropyLoss)
        >>> arcface_layer = create_metric_learning_criterion('arcface', 
        ...                                                 embedding_dim=512, 
        ...                                                 num_classes=163)
        >>> ce_loss = nn.CrossEntropyLoss()
    """
    loss_type_lower = loss_type.lower()
    
    if loss_type_lower in ['tripletloss', 'triplet']:
        margin = kwargs.get('margin', DEFAULT_TRIPLET_MARGIN)
        return TripletLoss(margin=margin)
        
    elif loss_type_lower in ['contrastiveloss', 'contrastive']:
        margin = kwargs.get('margin', DEFAULT_CONTRASTIVE_MARGIN)
        return ContrastiveLoss(margin=margin)
        
    elif loss_type_lower in ['arcfaceloss', 'arcface']:
        if embedding_dim is None or num_classes is None:
            raise ValueError("embedding_dim and num_classes son requeridos para ArcFace")
        
        scale = kwargs.get('scale', DEFAULT_ARCFACE_SCALE)
        margin = kwargs.get('margin', DEFAULT_ARCFACE_MARGIN)
        easy_margin = kwargs.get('easy_margin', False)
        
        return ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin
        )
        
    else:
        raise ValueError(f"Tipo de pérdida '{loss_type}' no soportado. "
                        "Use 'triplet', 'contrastive', o 'arcface'.")
