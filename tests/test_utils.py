"""
Test del módulo de utilidades.

Testa las funciones de JSON y otras utilidades básicas.
"""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from src.utils import (
    safe_json_dump, safe_json_dumps, 
    NumpyJSONEncoder, convert_numpy_keys
)


def test_numpy_json_encoder():
    """Test del encoder JSON para NumPy."""
    encoder = NumpyJSONEncoder()
    assert encoder is not None
    
    # Test con array numpy
    arr = np.array([1, 2, 3])
    result = encoder.encode(arr.tolist())
    assert isinstance(result, str)


def test_convert_numpy_keys():
    """Test conversión de keys NumPy."""
    data = {
        np.int64(1): 'value1',
        'normal_key': 'value2'
    }
    
    converted = convert_numpy_keys(data)
    assert isinstance(converted, dict)
    assert len(converted) == 2
    assert 'normal_key' in converted
    assert converted['normal_key'] == 'value2'


def test_safe_json_dumps():
    """Test serialización segura a string JSON."""
    data = {
        'array': np.array([1, 2, 3]),
        'normal': 'value',
        'number': 42
    }
    
    result = safe_json_dumps(data)
    assert isinstance(result, str)
    
    # Verificar que se puede parsear de vuelta
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    assert 'normal' in parsed
    assert parsed['normal'] == 'value'


def test_safe_json_dump():
    """Test guardado seguro a archivo JSON."""
    data = {
        'test': 'value',
        'number': 123,
        'array': [1, 2, 3]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Test guardado
        safe_json_dump(data, temp_path)
        
        # Verificar que el archivo existe y se puede leer
        path = Path(temp_path)
        assert path.exists()
        
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['test'] == 'value'
        assert loaded['number'] == 123
        assert loaded['array'] == [1, 2, 3]
        
    finally:
        # Limpiar
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])