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
    from src.data import DataFrameMaker, create_compcars_dataset
    from src.data import CarDataset, create_car_dataset
    assert DataFrameMaker is not None
    assert create_compcars_dataset is not None
    assert CarDataset is not None
    assert create_car_dataset is not None


def test_models_imports():
    """Test importación del módulo models."""
    from src.models import VisionModelError, MultiViewVisionModel, create_vision_model
    assert VisionModelError is not None
    assert MultiViewVisionModel is not None
    assert create_vision_model is not None


def test_utils_imports():
    """Test importación del módulo utils."""
    from src.utils import convert_numpy_keys, NumpyJSONEncoder, safe_json_dump, safe_json_dumps
    from src.utils import DimensionalityReducer
    from src.utils import fast_purity_calculation, ClusteringAnalyzer
    from src.utils import ClusterVisualizer
    assert convert_numpy_keys is not None
    assert NumpyJSONEncoder is not None
    assert safe_json_dump is not None
    assert safe_json_dumps is not None
    assert DimensionalityReducer is not None
    assert fast_purity_calculation is not None
    assert ClusteringAnalyzer is not None
    assert ClusterVisualizer is not None


def test_pipeline_imports():
    """Test importación del módulo pipeline."""
    from src.pipeline import FineTuningPipelineError, FineTuningPipeline, create_finetuning_pipeline
    from src.pipeline import EmbeddingsPipelineError, EmbeddingsPipeline, create_embeddings_pipeline
    assert FineTuningPipelineError is not None
    assert FineTuningPipeline is not None
    assert create_finetuning_pipeline is not None
    assert EmbeddingsPipelineError is not None
    assert EmbeddingsPipeline is not None
    assert create_embeddings_pipeline is not None


def test_main_src_imports():
    """Test importación directa desde src."""
    from src import DataFrameMaker, CarDataset, MultiViewVisionModel, DimensionalityReducer, ClusteringAnalyzer, ClusterVisualizer, FineTuningPipeline, EmbeddingsPipeline
    from src import create_compcars_dataset, create_car_dataset, create_vision_model, create_finetuning_pipeline, create_embeddings_pipeline
    assert DataFrameMaker is not None
    assert CarDataset is not None
    assert MultiViewVisionModel is not None
    assert DimensionalityReducer is not None
    assert ClusteringAnalyzer is not None
    assert ClusterVisualizer is not None
    assert FineTuningPipeline is not None
    assert EmbeddingsPipeline is not None
    assert create_compcars_dataset is not None
    assert create_car_dataset is not None
    assert create_vision_model is not None
    assert create_finetuning_pipeline is not None
    assert create_embeddings_pipeline is not None

    
if __name__ == "__main__":
    pytest.main([__file__])