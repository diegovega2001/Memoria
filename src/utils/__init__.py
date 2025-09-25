"""
Módulo de utilidades para el proyecto CompCars.

Este paquete contiene herramientas de análisis, visualización,
clustering y utilidades generales para el proyecto.

Classes:
    ClusteringAnalyzer: Análisis de clustering optimizado con múltiples algoritmos
    DimensionalityReducer: Reducción de dimensionalidad con PCA, t-SNE, UMAP
    ClusterVisualizer: Visualización de clusters y análisis de pureza
    NumpyJSONEncoder: Encoder JSON personalizado para arrays NumPy

Functions:
    fast_purity_calculation: Cálculo rápido de pureza de clustering
    convert_numpy_keys: Conversión de keys NumPy para serialización JSON
    safe_json_dump: Guardado seguro de datos en JSON
    safe_json_dumps: Serialización segura a string JSON
"""

from __future__ import annotations

# Importar las clases principales del módulo
from .ClusteringAnalyzer import ClusteringAnalyzer, fast_purity_calculation
from .DimensionalityReducer import DimensionalityReducer
from .ClusterVisualizer import ClusterVisualizer
from .JsonUtils import (
    NumpyJSONEncoder,
    convert_numpy_keys,
    safe_json_dump,
    safe_json_dumps,
)

# Definir qué se exporta cuando se hace "from src.utils import *"
__all__ = [
    # Analysis Classes
    'ClusteringAnalyzer',
    'DimensionalityReducer',
    'ClusterVisualizer',
    # JSON Utilities
    'NumpyJSONEncoder',
    'convert_numpy_keys',
    'safe_json_dump',
    'safe_json_dumps',
    # Analysis Functions
    'fast_purity_calculation',
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())