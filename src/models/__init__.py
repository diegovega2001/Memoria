"""
Módulo de modelos para el proyecto CompCars.

Este paquete contiene los modelos de deep learning para clasificación
de vehículos, enfocándose en modelos de visión multi-vista.

Classes:
    MultiViewVisionModel: Modelo de visión multi-vista con fine-tuning
    VisionModelError: Excepción para errores de modelos de visión

Functions:
    create_vision_model: Factory function para crear modelos de visión
"""

from __future__ import annotations

# Importar las clases principales del módulo
from .MyVisionModel import MultiViewVisionModel, VisionModelError, create_vision_model
from .Criterions import TripletLoss, ContrastiveLoss, ArcFaceLoss, ArcFaceInference, create_metric_learning_criterion
# Definir qué se exporta cuando se hace "from src.models import *"
__all__ = [
    # Clase base modelo
    'MultiViewVisionModel',
    # Clases metric learning
    'TripletLoss',
    'ContrastiveLoss',
    'ArcFaceLoss',
    'ArcFaceInference',
    # Exceptions
    'VisionModelError',
    # Factory functionszs
    'create_vision_model',
    'create_metric_learning_criterion'
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())