"""
Módulo de configuración para el proyecto CompCars.

Este paquete contiene clases y funciones para configurar transformaciones,
hiperparámetros y otros ajustes del proyecto.

Classes:
    TransformConfig: Configuración de transformaciones de imágenes.

Functions:
    create_standard_transform: Factory function para crear transformaciones estándar.
"""

from __future__ import annotations

from .TransformConfig import TransformConfig, create_standard_transform
__all__ = [
    # Base Class
    'TransformConfig',
    # Factory Function
    'create_standard_transform'
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())