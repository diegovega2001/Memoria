"""
Módulo de datos para el proyecto CompCars.

Este paquete contiene clases y funciones para el procesamiento y manejo
de datasets del proyecto CompCars, incluyendo generación de DataFrames
y datasets personalizados para PyTorch.

Classes:
    DataFrameMaker: Generador de dataset CSV a partir del CompCars dataset
    CarDataset: Dataset personalizado con splits por porcentaje para clasificación multi-vista
    MPerClassSamplerWrapper: Wrapper para MPerClassSampler de pytorch-metric-learning
    CompCarsDatasetError: Excepción para errores del dataset CompCars
    CarDatasetError: Excepción para errores del dataset de vehículos

Functions:
    create_compcars_dataset: Factory function para crear dataset CompCars
    create_car_dataset: Factory function para crear CarDataset con DataLoaders
    validate_dataset_structure: Validador de estructura de DataFrame
"""

from __future__ import annotations

# Importar las clases principales del módulo
from .DataFrameMaker import DataFrameMaker, CompCarsDatasetError, create_compcars_dataset
from .MyDataset import CarDataset, MPerClassSamplerWrapper, CarDatasetError, create_car_dataset, validate_dataset_structure


__all__ = [
    # Classes
    'DataFrameMaker',
    'CarDataset', 
    'MPerClassSamplerWrapper',
    # Exceptions
    'CompCarsDatasetError',
    'CarDatasetError',
    # Factory functions
    'create_compcars_dataset',
    'create_car_dataset',
    # Utilities
    'validate_dataset_structure',
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())