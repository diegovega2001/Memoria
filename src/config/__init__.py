"""
Módulo de configuración para el proyecto CompCars.

Este paquete contiene clases y funciones para configurar transformaciones,
hiperparámetros y otros ajustes del proyecto.

Classes:
    TransformConfig: Configuración de transformaciones de imágenes

Functions:
    create_standard_transform: Factory function para crear transformaciones estándar
"""

from __future__ import annotations

# Importar las clases principales del módulo
from .TransformConfig import TransformConfig, create_standard_transform
# Definir qué se exporta cuando se hace "from src.config import *"
__all__ = [
    # Clase base
    'TransformConfig',
    # Creador
    'create_standard_transform'
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging (opcional)
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())