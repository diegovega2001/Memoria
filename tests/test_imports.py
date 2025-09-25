"""
Test de importación de módulos del proyecto CompCars.

Este test verifica que todos los módulos se puedan importar
correctamente sin errores.
"""

import pytest


def test_config_imports():
    """Test importación del módulo config."""
    from src.config import TransformConfig, create_standard_transform
    assert TransformConfig is not None
    assert create_standard_transform is not None


def test_data_imports():
    """Test importación del módulo data."""
    from src.data import CarDataset, DataFrameMaker, create_car_dataset
    assert CarDataset is not None
    assert DataFrameMaker is not None
    assert create_car_dataset is not None


def test_models_imports():
    """Test importación del módulo models."""
    from src.models import MultiViewVisionModel, create_vision_model
    assert MultiViewVisionModel is not None
    assert create_vision_model is not None


def test_pipeline_imports():
    """Test importación del módulo pipeline."""
    from src.pipeline import FineTuningPipeline, EmbeddingsPipeline
    assert FineTuningPipeline is not None
    assert EmbeddingsPipeline is not None


def test_utils_imports():
    """Test importación del módulo utils."""
    from src.utils import ClusteringAnalyzer, DimensionalityReducer
    from src.utils import ClusterVisualizer, safe_json_dump
    assert ClusteringAnalyzer is not None
    assert DimensionalityReducer is not None
    assert ClusterVisualizer is not None
    assert safe_json_dump is not None


def test_main_src_imports():
    """Test importación directa desde src."""
    from src import CarDataset, MultiViewVisionModel, FineTuningPipeline
    from src import TransformConfig, create_car_dataset
    assert CarDataset is not None
    assert MultiViewVisionModel is not None
    assert FineTuningPipeline is not None
    assert TransformConfig is not None
    assert create_car_dataset is not None


if __name__ == "__main__":
    pytest.main([__file__])