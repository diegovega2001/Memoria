"""
Módulo de configuración de transformaciones para imágenes.

Este módulo proporciona una clase para configurar y aplicar transformaciones
de imágenes usando torchvision.
Incluye augmentación agresiva para mejorar la generalización.
"""

from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from torchvision import transforms

from src.defaults import (DEFAULT_COLOR_JITTER_BRIGHTNESS, DEFAULT_COLOR_JITTER_CONTRAST, DEFAULT_COLOR_JITTER_SATURATION, DEFAULT_COLOR_JITTER_HUE, DEFAULT_ROTATION_DEGREES, 
                          DEFAULT_RANDOM_ERASING_P, DEFAULT_GRAYSCALE, DEFAULT_RESIZE, DEFAULT_NORMALIZE, DEFAULT_USE_BBOX, DEFAULT_AUGMENT, IMAGENET_MEAN, IMAGENET_STD, 
                          GRAYSCALE_MEAN, GRAYSCALE_STD)


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class TransformConfig:
    """
    Configuración para transformaciones de imágenes.
    
    Esta clase permite configurar y generar transformaciones de PyTorch
    para preprocesamiento de imágenes, incluyendo redimensionado, 
    conversión a escala de grises, normalización, cropping por bounding box
    y augmentación agresiva para entrenamiento.
    
    Attributes:
        grayscale: Si True, convierte la imagen a escala de grises manteniendo
                  3 canales para compatibilidad con modelos preentrenados.
        resize: Tupla (height, width) para redimensionar la imagen. 
               Si es None, no se aplica redimensionado.
        normalize: Si True, aplica normalización usando estadísticas de ImageNet
                  o valores apropiados para escala de grises.
        use_bbox: Si True, permite el uso de bounding boxes para hacer crop.
                 Requiere que se pase la bbox como parámetro en __call__.
        augment: Si True, aplica augmentación agresiva (solo para entrenamiento).
    
    Example:
        >>> config = TransformConfig(grayscale=True, resize=(256, 256), use_bbox=True, augment=True)
        >>> bbox = [100, 100, 400, 400]  # [x_min, y_min, x_max, y_max]
        >>> processed_image = config(image, bbox=bbox)
    """
    grayscale: bool = DEFAULT_GRAYSCALE
    resize: Optional[Tuple[int, int]] = DEFAULT_RESIZE
    normalize: bool = DEFAULT_NORMALIZE
    use_bbox: bool = DEFAULT_USE_BBOX
    augment: bool = DEFAULT_AUGMENT
    
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
        
        # Augmentación Agresiva
        if self.augment:
            transform_list.extend([
                # Random horizontal flip (50% probabilidad)
                transforms.RandomHorizontalFlip(p=0.5),
                
                # Random rotation (±5 grados)
                transforms.RandomRotation(
                    degrees=DEFAULT_ROTATION_DEGREES,
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                
                # Random affine (zoom, traslación)
                transforms.RandomAffine(
                    degrees=0,  # Ya tenemos rotación arriba
                    translate=(0.05, 0.05),  # ±0.05% traslación
                    scale=(0.95, 1.05),  # 95%-115% zoom
                    shear=None,
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                
                # Color jitter (brillo, contraste, saturación, hue)
                transforms.ColorJitter(
                    brightness=DEFAULT_COLOR_JITTER_BRIGHTNESS,
                    contrast=DEFAULT_COLOR_JITTER_CONTRAST,
                    saturation=DEFAULT_COLOR_JITTER_SATURATION,
                    hue=DEFAULT_COLOR_JITTER_HUE
                ),
            ])
        
        # Conversión a tensor (siempre necesaria)
        transform_list.append(transforms.ToTensor())
        
        # Random erasing 
        if self.augment:
            transform_list.append(
                transforms.RandomErasing(
                    p=DEFAULT_RANDOM_ERASING_P,
                    scale=(0.02, 0.08),  # Tamaño del área a borrar
                    ratio=(0.5, 2.0),     # Aspect ratio
                    value='random'        # Valor aleatorio para el área borrada
                )
            )
        
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
            Imagen recortada si el bbox es válido, imagen original en caso contrario.
        """
        try:
            # Validar que tenemos una imagen
            if image is None:
                logging.warning("Imagen es None, no se puede aplicar bbox crop")
                return image
                
            # Parsear bbox si es string
            if isinstance(bbox, str):
                try:
                    bbox = ast.literal_eval(bbox)
                except (ValueError, SyntaxError) as e:
                    logging.warning(f"Error parseando bbox string '{bbox}': {e}, usando imagen completa")
                    return image
            
            # Validar formato
            if not isinstance(bbox, (list, tuple)):
                logging.warning(f"Bbox debe ser lista o tupla, recibido {type(bbox)}, usando imagen completa")
                return image
                
            if len(bbox) != 4:
                logging.warning(f"Bbox debe tener 4 coordenadas, recibido {len(bbox)} en {bbox}, usando imagen completa")
                return image
            
            # Extraer y validar coordenadas
            try:
                x_min, y_min, x_max, y_max = [float(x) for x in bbox]
            except (ValueError, TypeError) as e:
                logging.warning(f"Error convirtiendo coordenadas de bbox {bbox}: {e}, usando imagen completa")
                return image
            
            # Asegurar que las coordenadas estén dentro de la imagen ANTES de validar orden
            try:
                img_width, img_height = image.size
            except AttributeError:
                logging.warning(f"Imagen no tiene atributo 'size', tipo: {type(image)}, usando imagen completa")
                return image
            
            # Ajustar coordenadas a límites de la imagen
            x_min = max(0, min(x_min, img_width - 1))
            y_min = max(0, min(y_min, img_height - 1))
            x_max = max(x_min + 1, min(x_max, img_width))  # Asegurar al menos 1 pixel de ancho
            y_max = max(y_min + 1, min(y_max, img_height))  # Asegurar al menos 1 pixel de alto
            
            # Validar orden de coordenadas DESPUÉS de ajustar
            if x_min >= x_max or y_min >= y_max:
                logging.warning(f"Bbox con coordenadas inválidas después de ajustar (min >= max): original={bbox}, ajustado=[{x_min}, {y_min}, {x_max}, {y_max}], usando imagen completa")
                return image
            
            # Convertir a enteros
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            
            # Aplicar crop
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            return cropped_image
            
        except Exception as e:
            # Captura cualquier excepción inesperada
            logging.warning(f"Error inesperado aplicando bbox crop con bbox={bbox}: {type(e).__name__}: {e}, usando imagen completa")
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
            RuntimeError: Si hay un error crítico durante la transformación que
                         impide procesar la imagen (por ejemplo, imagen corrupta).
        """
        try:
            # Aplicar crop de bounding box si está habilitado y se proporciona
            # _apply_bbox_crop siempre retorna una imagen válida
            if self.use_bbox and bbox is not None:
                image = self._apply_bbox_crop(image, bbox)
            
            # Aplicar transformaciones de torchvision
            transform = self._get_transforms()
            transformed_image = transform(image)
            
            return transformed_image
            
        except Exception as e:
            # Log del error con más contexto para debugging
            error_msg = (
                f"Error crítico al aplicar transformaciones: {type(e).__name__}: {e}. "
                f"Configuración: grayscale={self.grayscale}, resize={self.resize}, "
                f"normalize={self.normalize}, use_bbox={self.use_bbox}, augment={self.augment}"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    
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
    grayscale: bool = DEFAULT_GRAYSCALE,
    use_bbox: bool = DEFAULT_USE_BBOX,
    augment: bool = DEFAULT_AUGMENT
) -> TransformConfig:
    """
    Crea una configuración estándar de transformaciones.
    
    Args:
        size: Tamaño de redimensionado (height, width).
        grayscale: Si aplicar escala de grises.
        use_bbox: Si habilitar el uso de bounding boxes para crop.
        augment: Si aplicar augmentación agresiva (para entrenamiento).
        
    Returns:
        TransformConfig: Configuración lista para usar.
    """
    return TransformConfig(
        grayscale=grayscale,
        resize=size,
        normalize=True,
        use_bbox=use_bbox,
        augment=augment
    )