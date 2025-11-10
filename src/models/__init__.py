"""
Módulo de modelos para el proyecto CompCars.

Este paquete contiene los modelos de deep learning para clasificación
de vehículos, enfocándose en modelos de visión multi-vista y metric learning
usando pytorch-metric-learning.

Classes:
    MultiViewVisionModel: Modelo de visión multi-vista con fine-tuning
    MultiViewCLIPModel: Modelo CLIP multi-vista para imagen-texto
    TripletLossWrapper: Wrapper para Triplet Loss de pytorch-metric-learning
    ContrastiveLossWrapper: Wrapper para Contrastive Loss de pytorch-metric-learning
    ArcFaceLossWrapper: Wrapper para ArcFace Loss de pytorch-metric-learning
    NTXentLossWrapper: Wrapper para NT-Xent Loss (InfoNCE)
    MultiSimilarityLossWrapper: Wrapper para MultiSimilarity Loss
    CLIPLoss: CLIP contrastive loss para imagen-texto
    VisionModelError: Excepción para errores de modelos de visión
    CLIPModelError: Excepción para errores de modelos CLIP

Functions:
    create_vision_model: Factory function para crear modelos de visión
    create_clip_model: Factory function para crear modelos CLIP
    create_metric_learning_loss: Factory function para crear pérdidas de metric learning
    create_miner: Factory function para crear miners de pytorch-metric-learning
"""

from __future__ import annotations

# Importar las clases principales del módulo
from .MyVisionModel import MultiViewVisionModel, VisionModelError, create_vision_model
from .MyCLIPModel import MultiViewCLIPModel, CLIPModelError, create_clip_model
from .MetricLearningLosses import (
    create_metric_learning_loss,
    TripletLossWrapper,
    ContrastiveLossWrapper,
    ArcFaceLossWrapper,
    NTXentLossWrapper,
    MultiSimilarityLossWrapper,
    CLIPLoss,
    create_miner
)

# Definir qué se exporta cuando se hace "from src.models import *"
__all__ = [
    # Clase base modelo vision
    'MultiViewVisionModel',
    # Clase base modelo CLIP
    'MultiViewCLIPModel',
    # Metric Learning Losses (pytorch-metric-learning wrappers)
    'TripletLossWrapper',
    'ContrastiveLossWrapper',
    'ArcFaceLossWrapper',
    'NTXentLossWrapper',
    'MultiSimilarityLossWrapper',
    'CLIPLoss',
    # Exceptions
    'VisionModelError',
    'CLIPModelError',
    # Factory functions
    'create_vision_model',
    'create_clip_model',
    'create_metric_learning_loss',
    'create_miner'
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())