"""
Utilidades para serialización JSON con tipos numpy y torch.

Este módulo proporciona herramientas para manejar la serialización de tipos
numpy y PyTorch a JSON, incluyendo conversión de claves y valores.
"""

import json
from typing import Any

import numpy as np
import torch


def convert_numpy_keys(obj: Any) -> Any:
    """
    Convierte recursivamente las claves numpy en diccionarios a strings.
    
    Args:
        obj: Objeto a convertir (puede ser dict, list, o cualquier tipo)
        
    Returns:
        Objeto con claves numpy convertidas a strings
    """
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, (np.integer, np.int64, np.int32, np.int16, np.int8)) else k: convert_numpy_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_numpy_keys(item) for item in obj]
    else:
        return obj


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Encoder JSON personalizado para manejar tipos de numpy y otros objetos no serializables.
    
    Maneja:
    - numpy.integer -> int
    - numpy.floating -> float  
    - numpy.ndarray -> list
    - torch.Tensor -> list (solo si es pequeño) o descripción de shape
    - Otros tipos numpy-like con método .item()
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            # Solo convertir tensores pequeños a lista para evitar JSON gigantes
            return obj.detach().cpu().numpy().tolist() if obj.numel() <= 1000 else str(obj.shape)
        elif hasattr(obj, 'item'):  # Para otros tipos numpy-like
            return obj.item()
        return super().default(obj)


def safe_json_dump(obj: Any, file_path: str, indent: int = 2) -> None:
    """
    Guarda un objeto a JSON manejando tipos numpy automáticamente.
    
    Args:
        obj: Objeto a serializar
        file_path: Ruta del archivo donde guardar
        indent: Indentación para el JSON (default: 2)
    """
    with open(file_path, "w") as f:
        json.dump(convert_numpy_keys(obj), f, indent=indent, cls=NumpyJSONEncoder)


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Convierte un objeto a string JSON manejando tipos numpy automáticamente.
    
    Args:
        obj: Objeto a serializar
        indent: Indentación para el JSON (default: 2)
        
    Returns:
        String JSON
    """
    return json.dumps(convert_numpy_keys(obj), indent=indent, cls=NumpyJSONEncoder)