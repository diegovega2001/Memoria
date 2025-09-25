"""
Test del módulo de configuración.

Testa la funcionalidad básica de TransformConfig y
la creación de transformaciones estándar.
"""

import pytest
from src.config import TransformConfig, create_standard_transform


def test_transform_config_creation():
    """Test creación básica de TransformConfig."""
    config = TransformConfig()
    assert config is not None
    assert hasattr(config, 'grayscale')
    assert hasattr(config, 'resize')
    assert hasattr(config, 'normalize')
    assert hasattr(config, 'use_bbox')


def test_transform_config_with_params():
    """Test creación de TransformConfig con parámetros."""
    config = TransformConfig(
        grayscale=True,
        resize=(224, 224),
        normalize=True,
        use_bbox=False
    )
    assert config.grayscale is True
    assert config.resize == (224, 224)
    assert config.normalize is True
    assert config.use_bbox is False


def test_create_standard_transform():
    """Test creación de transformación estándar."""
    transform = create_standard_transform()
    assert transform is not None
    
    # Test con parámetros
    transform_custom = create_standard_transform(
        size=(256, 256),
        grayscale=True
    )
    assert transform_custom is not None


if __name__ == "__main__":
    pytest.main([__file__])