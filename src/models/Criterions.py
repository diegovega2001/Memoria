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
    
    La función de pérdida Contrastive Loss trabaja con pares de embeddings,
    minimizando la distancia entre pares positivos (misma clase) y maximizando
    la distancia entre pares negativos (diferentes clases) hasta un margen.
    
    Loss = (1-Y) * 1/2 * (D^2) + (Y) * 1/2 * max(0, margin-D)^2
    donde Y=0 para pares positivos, Y=1 para pares negativos
    
    Args:
        margin: Margen mínimo para pares negativos.
        reduction: Tipo de reducción ('mean', 'sum', 'none').
    
    Example:
        >>> contrastive_loss = ContrastiveLoss(margin=1.0)
        >>> loss = contrastive_loss(embeddings1, embeddings2, labels)
    """
    
    def __init__(self, margin: float = DEFAULT_CONTRASTIVE_MARGIN, reduction: str = 'mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de Contrastive Loss.
        
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
