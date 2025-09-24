"""
Módulo de configuración de transformaciones para imágenes.

Este módulo proporciona una clase para configurar y aplicar transformaciones
de imágenes usando torchvision, siguiendo las mejores prácticas de Python.
"""

from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from torchvision import transforms

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Constantes para normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]

# Tamaño por defecto para ImageNet
DEFAULT_RESIZE = (224, 224)


@dataclass
class TransformConfig:
    """
    Configuración para transformaciones de imágenes.
    
    Esta clase permite configurar y generar transformaciones de PyTorch
    para preprocesamiento de imágenes, incluyendo redimensionado, 
    conversión a escala de grises, normalización y cropping por bounding box.
    
    Attributes:
        grayscale: Si True, convierte la imagen a escala de grises manteniendo
                  3 canales para compatibilidad con modelos preentrenados.
        resize: Tupla (height, width) para redimensionar la imagen. 
               Si es None, no se aplica redimensionado.
        normalize: Si True, aplica normalización usando estadísticas de ImageNet
                  o valores apropiados para escala de grises.
        use_bbox: Si True, permite el uso de bounding boxes para hacer crop.
                 Requiere que se pase la bbox como parámetro en __call__.
    
    Example:
        >>> config = TransformConfig(grayscale=True, resize=(256, 256), use_bbox=True)
        >>> bbox = [100, 100, 400, 400]  # [x_min, y_min, x_max, y_max]
        >>> processed_image = config(image, bbox=bbox)
    """
    
    grayscale: bool = False
    resize: Optional[Tuple[int, int]] = DEFAULT_RESIZE
    normalize: bool = True
    use_bbox: bool = False
    
    def __post_init__(self) -> None:
        """Valida los parámetros después de la inicialización."""
        if self.resize is not None:
            if (not isinstance(self.resize, (tuple, list)) or 
                len(self.resize) != 2 or 
                not all(isinstance(x, int) and x > 0 for x in self.resize)):
                raise ValueError(
                    "resize debe ser una tupla de dos enteros positivos (height, width)"
                )
    
    def _get_transforms(self) -> transforms.Compose:
        """
        Crea y retorna una composición de transformaciones.
        
        Returns:
            transforms.Compose: Composición de transformaciones configuradas.
            
        Raises:
            ValueError: Si los parámetros de configuración son inválidos.
        """
        transform_list = []
        
        # Conversión a escala de grises
        if self.grayscale:
            # Mantener 3 canales para compatibilidad con modelos preentrenados
            transform_list.append(transforms.Grayscale(num_output_channels=3))
        
        # Redimensionado
        if self.resize is not None:
            transform_list.append(transforms.Resize(self.resize))
        
        # Conversión a tensor (siempre necesaria)
        transform_list.append(transforms.ToTensor())
        
        # Normalización
        if self.normalize:
            if self.grayscale:
                transform_list.append(
                    transforms.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD)
                )
            else:
                transform_list.append(
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                )
        
        return transforms.Compose(transform_list)
    
    def _apply_bbox_crop(self, image: Any, bbox: Union[list, str]) -> Any:
        """
        Aplica crop usando bounding box.
        
        Args:
            image: Imagen PIL a recortar.
            bbox: Bounding box [x_min, y_min, x_max, y_max] como lista o string.
            
        Returns:
            Imagen recortada.
            
        Raises:
            ValueError: Si el formato de bbox es inválido.
        """
        try:
            # Parsear bbox si es string
            if isinstance(bbox, str):
                bbox = ast.literal_eval(bbox)
            
            # Validar formato
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"Bbox debe tener 4 coordenadas, recibido: {bbox}")
            
            x_min, y_min, x_max, y_max = bbox
            
            # Validar coordenadas
            if x_min >= x_max or y_min >= y_max:
                logging.warning(f"Bbox inválida (coordenadas): {bbox}, usando imagen completa")
                return image
            
            # Asegurar que las coordenadas estén dentro de la imagen
            img_width, img_height = image.size
            x_min = max(0, min(x_min, img_width - 1))
            y_min = max(0, min(y_min, img_height - 1))
            x_max = max(x_min + 1, min(x_max, img_width))
            y_max = max(y_min + 1, min(y_max, img_height))
            
            # Aplicar crop
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            return cropped_image
            
        except Exception as e:
            logging.warning(f"Error aplicando bbox crop {bbox}: {e}, usando imagen completa")
            return image
    
    def __call__(self, image: Any, bbox: Optional[Union[list, str]] = None) -> Any:
        """
        Aplica las transformaciones configuradas a una imagen.
        
        Args:
            image: Imagen a transformar (PIL Image, tensor, etc.).
            bbox: Bounding box opcional para hacer crop [x_min, y_min, x_max, y_max].
                 Puede ser una lista o string con formato "[x1, y1, x2, y2]".
            
        Returns:
            Any: Imagen transformada.
            
        Raises:
            RuntimeError: Si hay un error durante la transformación.
        """
        try:
            # Aplicar crop de bounding box si está habilitado y se proporciona
            if self.use_bbox and bbox is not None:
                image = self._apply_bbox_crop(image, bbox)
            
            transform = self._get_transforms()
            return transform(image)
        except Exception as e:
            raise RuntimeError(f"Error al aplicar transformaciones: {e}") from e
    
    def __repr__(self) -> str:
        """Representación string mejorada para debugging."""
        return (
            f"{self.__class__.__name__}("
            f"grayscale={self.grayscale}, "
            f"resize={self.resize}, "
            f"normalize={self.normalize}, "
            f"use_bbox={self.use_bbox})"
        )


# Funciones de conveniencia
def create_standard_transform(
    size: Tuple[int, int] = DEFAULT_RESIZE,
    grayscale: bool = False,
    use_bbox: bool = False
) -> TransformConfig:
    """
    Crea una configuración estándar de transformaciones.
    
    Args:
        size: Tamaño de redimensionado (height, width).
        grayscale: Si aplicar escala de grises.
        use_bbox: Si habilitar el uso de bounding boxes para crop.
        
    Returns:
        TransformConfig: Configuración lista para usar.
    """
    return TransformConfig(
        grayscale=grayscale,
        resize=size,
        normalize=True,
        use_bbox=use_bbox
    )


def create_inference_transform(use_bbox: bool = False) -> TransformConfig:
    """
    Crea una configuración optimizada para inferencia.
    
    Args:
        use_bbox: Si habilitar el uso de bounding boxes para crop.
    
    Returns:
        TransformConfig: Configuración para inferencia.
    """
    return TransformConfig(
        grayscale=False,
        resize=DEFAULT_RESIZE,
        normalize=True,
        use_bbox=use_bbox
    )