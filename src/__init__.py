"""
Proyecto CompCars - Análisis de Vehículos con Deep Learning.

Este proyecto implementa modelos de visión computacional para clasificación
de vehículos utilizando el dataset CompCars, con enfoque en fine-tuning,
análisis de embeddings, clustering y modelos multimodales.

Paquetes:
    config: Configuración de transformaciones y hiperparámetros
    data: Procesamiento y manejo de datasets
    models: Modelos de visión y multimodales
    pipeline: Pipelines de entrenamiento y análisis
    utils: Utilidades de análisis y visualización

Características principales:
    - Fine-tuning de modelos pre-entrenados (ResNet, etc.)
    - Análisis de embeddings antes y después del fine-tuning
    - Clustering y visualización de representaciones aprendidas
    - Modelos multimodales (visión + texto) con CLIP
    - Soporte multi-vista (front/rear) de vehículos
    - Pipelines reproducibles con configuración JSON
"""

from __future__ import annotations

# Importar los submódulos principales
from . import config
from . import data
from . import models
from . import utils
from . import pipeline

# Importar todo lo utilizable desde el acceso directo
from .config import TransformConfig, create_standard_transform
from .data import DataFrameMaker, create_compcars_dataset, CarDataset, create_car_dataset
from .models import MultiViewVisionModel, create_vision_model
from .utils import DimensionalityReducer, ClusteringAnalyzer, ClusterVisualizer
from .pipeline import FineTuningPipeline, create_embeddings_pipeline, EmbeddingsPipeline, create_finetuning_pipeline

# Definir qué se exporta cuando se hace "from src import *"
__all__ = [
    # Submodules
    'config',
    'data', 
    'models',
    'utils',
    'pipeline',
    # Main classes (shortcuts)
    'TransformConfig',
    'DataFrameMaker',
    'CarDataset',
    'MultiViewVisionModel',
    'DimensionalityReducer',
    'ClusteringAnalyzer',
    'ClusterVisualizer',
    'FineTuningPipeline',
    'EmbeddingsPipeline',
    # Factory functions
    'create_standard_transform',
    'create_compcars_dataset',
    'create_car_dataset',
    'create_vision_model',
    'create_embeddings_pipeline',
    'create_finetuning_pipeline'
]

# Información del proyecto
__version__ = '0.1.0'
__author__ = 'Diego Vega'
__email__ = 'diego.vega@example.com'
__description__ = 'Análisis de vehículos con deep learning usando dataset CompCars'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())