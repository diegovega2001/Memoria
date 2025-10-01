"""
Módulo de pipelines para el proyecto CompCars.

Este paquete contiene los pipelines principales para fine-tuning
y análisis de embeddings del proyecto CompCars.

Classes:
    FineTuningPipeline: Pipeline completo para fine-tuning con evaluación
    EmbeddingsPipeline: Pipeline para análisis de embeddings y clustering
    FineTuningPipelineError: Excepción para errores del pipeline de fine-tuning
    EmbeddingsPipelineError: Excepción para errores del pipeline de embeddings

Functions:
    create_finetuning_pipeline: Factory function para crear pipeline de fine-tuning
    create_embeddings_pipeline: Factory function para crear pipeline de embeddings
"""

from __future__ import annotations

# Importar las clases principales del módulo
from .FineTuningPipeline import (
    FineTuningPipeline,
    FineTuningPipelineError,
    create_finetuning_pipeline,
)
from .EmbeddingsPipeline import (
    EmbeddingsPipeline,
    EmbeddingsPipelineError,
    create_embeddings_pipeline,
)

# Definir qué se exporta cuando se hace "from src.pipeline import *"
__all__ = [
    # Clases de pipeline
    'FineTuningPipeline',
    'EmbeddingsPipeline',
    # Exceptions
    'FineTuningPipelineError',
    'EmbeddingsPipelineError',
    # Funciones de generación
    'create_finetuning_pipeline',
    'create_embeddings_pipeline',
]

# Información del módulo
__version__ = '0.1.0'
__author__ = 'Diego Vega'

# Logging para debugging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())