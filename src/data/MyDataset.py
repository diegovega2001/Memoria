"""
Módulo de dataset personalizado para clasificación de vehículos multi-vista con splits por porcentaje.

Este módulo proporciona clases para manejar datasets de vehículos con múltiples
vistas, usando splits tradicionales por porcentaje y manejo robusto de clases long-tail.
"""

from __future__ import annotations

import copy
import logging
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, default_collate
from pytorch_metric_learning.samplers import MPerClassSampler

from src.defaults import (
    DEFAULT_VIEWS, DEFAULT_MIN_IMAGES, DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO, DEFAULT_SEED, 
    DEFAULT_TRANSFORM, DEFAULT_DESCRIPTION_INCLUDE, DEFAULT_ONESHOT, DEFAULT_ONESHOT_RATIO, DEFAULT_P, 
    DEFAULT_K, DEFAULT_MODEL_TYPE,DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, MODEL_TYPES, DESCRIPTION_OPTIONS, 
    ERROR_INVALID_DESCRIPTION, ERROR_INVALID_MODEL_TYPE, CLASS_GRANULARITY_OPTIONS, DEFAULT_CLASS_GRANULARITY
)

from src.config.TransformConfig import create_standard_transform

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CarDatasetError(Exception):
    """Excepción personalizada para errores del dataset de vehículos."""
    pass


class CarDataset(Dataset):
    """
    Dataset personalizado para clasificación de vehículos multi-vista con splits por porcentaje.

    Esta clase maneja datasets de vehículos con múltiples puntos de vista,
    usando splits tradicionales (train/val/test) basados en porcentajes.
    Maneja robustamente distribuciones long-tail separando clases regulares de one-shot.
    
    Estrategia de Clases:
    - Clases regulares (≥ min_images): Usadas en train/val/test
    - Clases one-shot (< min_images): Solo en test para evaluar generalización
    - TODAS las clases tienen labels encodificados (para clustering/evaluación)
    
    Estrategia de Descripciones Textuales:
    - Train: Descripción detallada según 'description_include' (full/model/etc.)
    - Val/Test: Descripción parcial (solo make + type) para simular zero-shot
    
    Reproducibilidad:
    - El parámetro 'seed' garantiza reproducibilidad total
    - Con el mismo seed y configuración, se generará el mismo dataset
    - Los splits train/val/test serán idénticos entre ejecuciones

    Attributes:
        df: DataFrame con los datos del dataset.
        views: Lista de vistas/viewpoints a incluir.
        num_views: Número de vistas configuradas.
        class_granularity: 'model' o 'model+year'.
        min_images: Mínimo de imágenes para considerar una clase en train/val/test.
        train_ratio: Proporción de datos para entrenamiento.
        val_ratio: Proporción de datos para validación.
        test_ratio: Proporción de datos para prueba.
        seed: Semilla para reproducibilidad.
        transform: Transformaciones a aplicar a las imágenes.
        model_type: Tipo de modelo ('vision', 'textual', 'both').
        description_include: Nivel de detalle en descripciones textuales (solo train).
        include_oneshot_in_test: Si agregar clases one-shot al conjunto de test.
        regular_classes: Clases con suficientes imágenes para splits.
        oneshot_classes: Clases con pocas imágenes (< min_images).
        label_encoder: Encoder para TODAS las clases (regulares + oneshot).
        train_samples: Muestras de entrenamiento (solo clases regulares).
        val_samples: Muestras de validación (solo clases regulares).
        test_samples: Muestras de prueba (regulares + one-shot si aplica).
        oneshot_samples: Muestras one-shot (para referencia).
        current_split: Split actual ('train', 'val', 'test').
    """
    def __init__(
        self,
        df: pd.DataFrame,
        views: List[str] = DEFAULT_VIEWS,
        class_granularity: str = DEFAULT_CLASS_GRANULARITY,
        min_images: int = DEFAULT_MIN_IMAGES,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        seed: int = DEFAULT_SEED,
        transform: Optional[Any] = DEFAULT_TRANSFORM,
        model_type: str = DEFAULT_MODEL_TYPE,
        description_include: str = DEFAULT_DESCRIPTION_INCLUDE,
        include_oneshot_in_test: bool = DEFAULT_ONESHOT,
        oneshot_ratio: Optional[float] = DEFAULT_ONESHOT_RATIO
    ) -> None:
        """
        Inicializa el dataset de vehículos con splits por porcentaje.

        Args:
            df: DataFrame con columnas: 'model', 'released_year', 'viewpoint', 'image_path', 'make', 'type'.
            views: Lista de viewpoints a incluir.
            class_granularity: 'model' (agrupa años) o 'model+year' (separa años).
            min_images: Mínimo de imágenes para que una clase entre en train/val/test (default: 6).
            train_ratio: Proporción para train (default: 0.7).
            val_ratio: Proporción para val (default: 0.2).
            test_ratio: Proporción para test (default: 0.1).
            seed: Semilla para reproducibilidad.
            transform: Transformaciones de torchvision.
            model_type: Tipo de salida ('vision', 'textual', 'both').
            description_include: Nivel de detalle en descripciones.
            include_oneshot_in_test: Si agregar clases one-shot a test (default: True).
            oneshot_ratio: Proporción de clases one-shot a incluir en test (default: 1.0 = todas).

        Raises:
            CarDatasetError: Si hay errores de configuración o datos.
            ValueError: Si los parámetros son inválidos.
        """
        # Validación de parámetros
        self._validate_parameters(
            df, views, class_granularity, min_images, 
            train_ratio, val_ratio, test_ratio, 
            model_type, description_include
        )

        # Configuración básica
        self.df = df.copy()
        self.views = views if views is not None else DEFAULT_VIEWS.copy()
        self.num_views = len(self.views)
        self.seed = seed
        self.class_granularity = class_granularity
        self.min_images = min_images
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform = transform
        self.model_type = model_type
        self.description_include = description_include
        self.include_oneshot_in_test = include_oneshot_in_test
        self.oneshot_ratio = oneshot_ratio if oneshot_ratio is not None else 1.0
        self.current_split = None

        # Validar que los ratios sumen 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Los ratios deben sumar 1.0, suma actual: {total_ratio}")

        # Configurar random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        logging.info(f"Inicializando CarDataset con {len(self.df)} registros")
        logging.info(f"Granularidad de clase: {self.class_granularity}")
        logging.info(f"Vistas configuradas: {self.views}")
        logging.info(f"Mínimo de imágenes por clase: {self.min_images}")
        logging.info(f"Splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        logging.info(f"One-shot en test: {'Sí' if self.include_oneshot_in_test else 'No'}")

        # Inicialización de componentes
        self._identify_classes()
        
        # Inicializar encoders 
        self.label_encoder = self._initialize_label_encoder()
        self.df = self._filter_dataframe()

        # Creación de splits
        self._create_data_splits()
        
        logging.info(f"Samples - Train: {len(self.train_samples)}, Val: {len(self.val_samples)}, Test: {len(self.test_samples)}")
        logging.info(f"One-shot classes: {len(self.oneshot_classes)} ({len(self.oneshot_samples)} samples)")

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        views: Optional[List[str]],
        class_granularity: str,
        min_images: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        model_type: str,
        description_include: str
    ) -> None:
        """Valida los parámetros de entrada."""
        # Validar DataFrame
        required_columns = {'model', 'released_year', 'viewpoint', 'image_path', 'make'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise CarDatasetError(f"Columnas faltantes en DataFrame: {missing}")
        
        if df.empty:
            raise CarDatasetError("El DataFrame no puede estar vacío")
        
        # Validar granularidad de clase
        if class_granularity not in CLASS_GRANULARITY_OPTIONS:
            raise ValueError(f"class_granularity debe ser uno de {CLASS_GRANULARITY_OPTIONS}")
        
        # Validar parámetros numéricos
        if min_images < 2:
            raise ValueError("min_images debe ser ≥2")
        
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio debe estar entre 0 y 1")
        if not (0 < val_ratio < 1):
            raise ValueError("val_ratio debe estar entre 0 y 1")
        if not (0 < test_ratio < 1):
            raise ValueError("test_ratio debe estar entre 0 y 1")
        
        # Validar opciones categóricas
        if model_type not in MODEL_TYPES:
            raise ValueError(ERROR_INVALID_MODEL_TYPE.format(MODEL_TYPES))
        
        if description_include not in DESCRIPTION_OPTIONS:
            raise ValueError(ERROR_INVALID_DESCRIPTION.format(DESCRIPTION_OPTIONS))
        
        # Validar vistas
        if views is not None:
            if not views or not isinstance(views, list):
                raise ValueError("views debe ser una lista no vacía")
            available_views = set(df['viewpoint'].unique())
            invalid_views = set(views) - available_views
            if invalid_views:
                raise CarDatasetError(
                    f"Views no disponibles en el dataset: {invalid_views}. "
                    f"Disponibles: {available_views}"
                )

    def _identify_classes(self) -> None:
        """
        Identifica clases regulares (≥ min_images) y one-shot (< min_images).
        
        Las clases regulares se usarán para train/val/test.
        Las clases one-shot se reservan para evaluar generalización en test.
        """
        logging.info(f"Identificando clases con granularidad '{self.class_granularity}'...")

        # Determinar columnas de agrupación
        if self.class_granularity == 'model+year':
            group_cols = ['model', 'released_year']
        else: 
            group_cols = ['model']
        
        # Añadir viewpoint para contar por vista
        group_cols_with_view = group_cols + ['viewpoint']
        
        # Contar imágenes por clase y vista
        counts = self.df.groupby(group_cols_with_view).size().unstack(fill_value=0)
        available_views = [v for v in self.views if v in counts.columns]

        if not available_views:
            raise CarDatasetError(f"Ninguna de las vistas especificadas existe en el dataset: {self.views}")
        
        # Total de imágenes por clase (sumando todas las vistas)
        total_counts = counts[available_views].sum(axis=1)

        # Filtrar clases que tienen al menos 1 imagen en todas las vistas requeridas
        valid_classes = []
        for class_key in total_counts.index:
            if self.class_granularity == 'model+year':
                class_views = counts.loc[class_key, available_views]
            else:
                class_views = counts.loc[class_key, available_views]
            
            # Verificar que tenga ≥1 imagen en cada vista
            if all(class_views >= 1):
                valid_classes.append(class_key)
        
        logging.info(f"Clases con ≥1 imagen en todas las vistas: {len(valid_classes)}")

        # Separar en regular vs one-shot
        # IMPORTANTE: min_images se refiere a mínimo POR VISTA, no total
        self.regular_classes = []
        self.oneshot_classes = []
        
        for class_key in valid_classes:
            class_views = counts.loc[class_key, available_views]
            # Verificar que CADA vista tenga al menos min_images
            if all(class_views >= self.min_images):
                self.regular_classes.append(class_key)
            else:
                self.oneshot_classes.append(class_key)
        
        logging.info(f"Clases regulares (≥{self.min_images} imgs): {len(self.regular_classes)}")
        logging.info(f"Clases one-shot (<{self.min_images} imgs): {len(self.oneshot_classes)}")

        if not self.regular_classes:
            raise CarDatasetError(
                f"No hay clases con al menos {self.min_images} imágenes en todas las vistas. "
                f"Considere reducir min_images o agregar más datos."
            )

    def _initialize_label_encoder(self) -> LabelEncoder:
        """
        Inicializa dos encoders de etiquetas:
        1. label_encoder_full: Todas las clases (regulares + oneshot) para evaluación
        2. label_encoder_training: Solo clases regulares para entrenamiento
        """
        # Encoder completo (para evaluación/clustering)
        label_encoder_full = LabelEncoder()
        all_classes = self.regular_classes + self.oneshot_classes
        
        if self.class_granularity == 'model+year':
            class_strings_full = [f"{model}_{year}" for model, year in all_classes]
        else: 
            class_strings_full = list(all_classes)
        
        label_encoder_full.fit(class_strings_full)
        
        # Encoder solo para entrenamiento (solo clases regulares)
        label_encoder_training = LabelEncoder()
        
        if self.class_granularity == 'model+year':
            class_strings_training = [f"{model}_{year}" for model, year in self.regular_classes]
        else: 
            class_strings_training = list(self.regular_classes)
        
        label_encoder_training.fit(class_strings_training)

        logging.info(f"LabelEncoder completo: {len(all_classes)} clases totales "
                    f"({len(self.regular_classes)} regulares + {len(self.oneshot_classes)} oneshot)")
        logging.info(f"LabelEncoder entrenamiento: {len(self.regular_classes)} clases regulares")

        # Guardar ambos encoders
        self.label_encoder_full = label_encoder_full
        return label_encoder_training  

    def _filter_dataframe(self) -> pd.DataFrame:
        """Filtra el DataFrame para incluir solo clases y vistas válidas."""
        initial_size = len(self.df)

        # Combinar clases regulares y one-shot
        all_valid_classes = set(self.regular_classes + self.oneshot_classes)
        
        df_filtered = self.df.copy()
        
        if self.class_granularity == 'model+year':
            df_filtered['class_key'] = list(zip(df_filtered['model'], df_filtered['released_year']))
        else:
            df_filtered['class_key'] = df_filtered['model']

        # Filtrado
        filtered_df = df_filtered[
            (df_filtered['class_key'].isin(all_valid_classes)) &
            (df_filtered['viewpoint'].isin(self.views))
        ].drop('class_key', axis=1)

        final_size = len(filtered_df)
        logging.info(f"DataFrame filtrado: {initial_size} → {final_size} registros")

        return filtered_df

    def _create_text_descriptor(self, image_paths: Union[str, List[str]]) -> str:
        """
        Crea descriptor textual para una imagen o par de imágenes.
        
        Usa self.current_split para determinar el nivel de detalle:
        - Train: Descripción detallada según self.description_include
        - Val/Test/One-Shot: Descripción parcial (solo make + type)

        Args:
            image_paths: Ruta(s) de imagen(es) para describir.

        Returns:
            Descripción textual del vehículo.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        try:
            rows = []
            for path in image_paths:
                matching_rows = self.df[self.df['image_path'] == path]
                if matching_rows.empty:
                    raise CarDatasetError(f"Imagen no encontrada en dataset: {path}")
                rows.append(matching_rows.iloc[0])
        except Exception as e:
            raise CarDatasetError(f"Error obteniendo información de imagen: {e}") from e
        
        # Información base
        make = rows[0]['make'].strip()
        type_str = rows[0].get('type', '').strip()
        model = rows[0]['model'].strip()
        year = str(rows[0]['released_year']).strip()
        viewpoints = [row['viewpoint'] for row in rows]

        # Construir descripción base con viewpoints
        if len(viewpoints) == 1:
            view_prefix = f"The {viewpoints[0]} view image of a"
        else:
            viewpoint_text = " and ".join(viewpoints)
            view_prefix = f"The {viewpoint_text} view images of a"
        
        # Val/Test/One-Shot: siempre descripción parcial (solo make + type)
        if self.current_split in ['val', 'test', 'oneshot']:
            if type_str:
                desc = f"{view_prefix} {make} vehicle type {type_str}."
            else:
                desc = f"{view_prefix} {make} vehicle."
        
        # Train: descripción según configuración description_include
        elif self.current_split == 'train':
            if self.description_include == 'full':
                if type_str:
                    desc = f"{view_prefix} {make} {model} vehicle from {year} type {type_str}."
                else:
                    desc = f"{view_prefix} {make} {model} vehicle from {year}."
                    
            elif self.description_include == 'model':
                if type_str:
                    desc = f"{view_prefix} {make} {model} vehicle type {type_str}."
                else:
                    desc = f"{view_prefix} {make} {model} vehicle."
            
            elif self.description_include == 'make':
                if type_str:
                    desc = f"{view_prefix} {make} vehicle type {type_str}."
                else:   
                    desc = f"{view_prefix} {make} vehicle."
            else:
                if type_str:   
                    desc = f"{view_prefix} vehicle type {type_str}."
                else:   
                    desc = f"{view_prefix} vehicle."
        
        else:
            raise ValueError(f"Split inválido: {self.current_split}")
        
        return desc

    def _create_data_splits(self) -> None:
        """Crea las divisiones train/val/test por porcentaje."""
        logging.info(f"Creando splits por porcentaje...")

        # Procesar clases regulares
        all_samples = []
        for class_key in self.regular_classes:
            samples = self._create_samples_for_class(class_key)
            all_samples.extend(samples)
        
        # Shuffle para asegurar mezcla
        random.shuffle(all_samples)
        
        # Calcular índices de corte
        n_total = len(all_samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # Split
        self.train_samples = all_samples[:n_train]
        self.val_samples = all_samples[n_train:n_train + n_val]
        self.test_samples = all_samples[n_train + n_val:]
        
        # Procesar clases one-shot
        self.oneshot_samples = []
        for class_key in self.oneshot_classes:
            samples = self._create_samples_for_class(class_key)
            self.oneshot_samples.extend(samples)
        
        # Añadir one-shot a test si está configurado
        if self.include_oneshot_in_test and self.oneshot_samples:
            # Seleccionar proporción de one-shot
            n_oneshot = int(len(self.oneshot_samples) * self.oneshot_ratio)
            selected_oneshot = random.sample(self.oneshot_samples, n_oneshot)
            self.test_samples.extend(selected_oneshot)
            logging.info(f"Añadidas {n_oneshot} muestras one-shot a test")
    
    def _create_samples_for_class(self, class_key: Union[Tuple[str, Any], str]) -> List:
        """Crea muestras para una clase específica."""
        # Filtrar data según granularidad
        if self.class_granularity == 'model+year':
            model, year = class_key
            class_data = self.df[
                (self.df['model'] == model) & 
                (self.df['released_year'] == year) &
                (self.df['viewpoint'].isin(self.views))
            ]
        else:  # 'model'
            model = class_key
            class_data = self.df[
                (self.df['model'] == model) &
                (self.df['viewpoint'].isin(self.views))
            ]
        
        # Agrupar por vista
        grouped = class_data.groupby('viewpoint')
        view_images = {}
        for view in self.views:
            if view in grouped.groups:
                paths = list(grouped.get_group(view)['image_path'])
                random.shuffle(paths)
                view_images[view] = paths
            else:
                view_images[view] = []
        
        # Crear samples
        min_images = min(len(view_images[view]) for view in self.views if len(view_images[view]) > 0)
        if min_images == 0:
            return []
        
        samples = []
        for i in range(min_images):
            image_pair = [view_images[view][i] for view in self.views if i < len(view_images[view])]
            if len(image_pair) == len(self.views):
                sample = (class_key, image_pair)
                samples.append(sample)
        
        return samples

    def set_split(self, split: str) -> None:
        """
        Establece el split actual del dataset.

        Args:
            split: Split a activar ('train', 'val', 'test').
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split debe ser 'train', 'val' o 'test', recibido: {split}")
        
        self.current_split = split
        logging.debug(f"Dataset switch to {split} split")

    def get_num_classes_for_training(self) -> int:
        """
        Retorna el número de clases REGULARES para entrenamiento.
        
        Esta es la cantidad de clases que deben usarse para la capa de salida
        del modelo durante el entrenamiento, ya que solo las clases regulares
        tienen suficientes muestras para train/val.
        
        Returns:
            Número de clases regulares.
        """
        return len(self.regular_classes)
    
    def get_total_num_classes(self) -> int:
        """
        Retorna el número TOTAL de clases (regulares + oneshot).
        
        Útil para evaluación de clustering y métricas que consideran
        todas las clases del dataset.
        
        Returns:
            Número total de clases.
        """
        return len(self.regular_classes) + len(self.oneshot_classes)
    
    def get_label_encoder_for_evaluation(self) -> LabelEncoder:
        """
        Retorna el label encoder completo (todas las clases) para evaluación/clustering.
        
        Este encoder debe usarse cuando se evalúan embeddings o se hace clustering
        incluyendo las clases oneshot.
        
        Returns:
            LabelEncoder con todas las clases (regulares + oneshot).
        """
        return self.label_encoder_full

    def get_current_samples(self) -> List:
        """Retorna las muestras del split actual."""
        if self.current_split == 'train':
            return self.train_samples
        elif self.current_split == 'val':
            return self.val_samples
        elif self.current_split == 'test':
            return self.test_samples
        else:
            raise ValueError(f"Split inválido: {self.current_split}")

    def __len__(self) -> int:
        """Retorna el número de muestras en el split actual."""
        return len(self.get_current_samples())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Obtiene una muestra del dataset.

        Args:
            idx: Índice de la muestra.

        Returns:
            Diccionario con la muestra procesada.
        """
        try:
            return self._load_sample(idx)
        except Exception as e:
            logging.error(f"Error cargando muestra {idx}: {e}")
            return None
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Carga y procesa una muestra individual."""
        samples = self.get_current_samples()
        sample_data = samples[idx]
        
        # Extraer componentes: ahora solo (class_key, image_paths)
        class_key, image_paths = sample_data
        
        # Obtener label
        if self.class_granularity == 'model+year':
            model, year = class_key
            label_str = f"{model}_{year}"
        else:
            label_str = class_key
        
        # Encode label según el split y si la clase es oneshot
        # Train/Val: Solo clases regulares (encoder de entrenamiento)
        # Test: Si include_oneshot_in_test y es oneshot, usar encoder completo
        if class_key in self.regular_classes:
            # Clase regular: siempre usar encoder de entrenamiento (0 a num_regular_classes-1)
            label = self.label_encoder.transform([label_str])[0]
        else:
            # Clase oneshot: solo válido en test, usar encoder completo
            if self.current_split == 'test' and self.include_oneshot_in_test:
                label = self.label_encoder_full.transform([label_str])[0]
            else:
                # No debería llegar aquí, pero por seguridad
                raise ValueError(f"Clase oneshot {class_key} encontrada fuera de test split")
        
        # Cargar imágenes
        images = []
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    if self.transform:
                        img = self.transform(img)
                    
                    images.append(img)
            except Exception as e:
                logging.error(f"Error cargando imagen {img_path}: {e}")
                raise
        
        # Stack imágenes
        if len(images) == 1:
            images_tensor = images[0]
        else:
            images_tensor = torch.stack(images)
        
        # Construir sample dict
        sample = {
            'images': images_tensor,
            'labels': torch.tensor(label, dtype=torch.long),
            'class_key': class_key,
            'image_paths': image_paths
        }
    
        # Añadir descripción textual si aplica
        if self.model_type in ['textual', 'multimodal']:
            text_desc = self._create_text_descriptor(image_paths)
            sample['text_description'] = text_desc
        
        return sample

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del dataset."""
        stats = {
            'total_samples': len(self.train_samples) + len(self.val_samples) + len(self.test_samples),
            'num_training_classes': len(self.regular_classes),  # Para la capa de salida del modelo
            'num_total_classes': len(self.regular_classes) + len(self.oneshot_classes),  # Para evaluación
            'num_regular_classes': len(self.regular_classes),
            'num_oneshot_classes': len(self.oneshot_classes),
            'num_views': self.num_views,
            'views': self.views,
            'class_granularity': self.class_granularity,
            'min_images': self.min_images,
            'seed': self.seed,
            'description_strategy': {
                'train': f'detailed ({self.description_include})',
                'val': 'partial (make + type)',
                'test': 'partial (make + type)'
            },
            'splits': {
                'train': {
                    'samples': len(self.train_samples),
                    'ratio': self.train_ratio,
                    'classes': 'regular only'
                },
                'val': {
                    'samples': len(self.val_samples),
                    'ratio': self.val_ratio,
                    'classes': 'regular only'
                },
                'test': {
                    'samples': len(self.test_samples),
                    'ratio': self.test_ratio,
                    'includes_oneshot': self.include_oneshot_in_test,
                    'oneshot_samples': len(self.oneshot_samples) if self.include_oneshot_in_test else 0,
                    'classes': 'regular + oneshot' if self.include_oneshot_in_test else 'regular only'
                }
            }
        }
        
        return stats

    def __str__(self) -> str:
        """Representación string del dataset."""
        stats = self.get_dataset_statistics()
        lines = []
        lines.append("="*70)
        lines.append(f"CarDataset Summary")
        lines.append("="*70)
        lines.append(f"Classes: {stats['num_training_classes']} training | "
                    f"{stats['num_total_classes']} total (+ {stats['num_oneshot_classes']} oneshot)")
        lines.append(f"Granularity: {stats['class_granularity']}")
        lines.append(f"Views: {stats['views']} (n={stats['num_views']})")
        lines.append(f"Min images per class: {stats['min_images']}")
        lines.append(f"Seed: {stats['seed']} (reproducible)")
        lines.append(f"\nText Descriptions:")
        lines.append(f"  Train: {stats['description_strategy']['train']}")
        lines.append(f"  Val:   {stats['description_strategy']['val']}")
        lines.append(f"  Test:  {stats['description_strategy']['test']}")
        lines.append(f"\nData Splits:")
        lines.append(f"  Train:  {stats['splits']['train']['samples']:6d} samples "
                    f"({stats['splits']['train']['ratio']:.1%}) - {stats['splits']['train']['classes']}")
        lines.append(f"  Val:    {stats['splits']['val']['samples']:6d} samples "
                    f"({stats['splits']['val']['ratio']:.1%}) - {stats['splits']['val']['classes']}")
        lines.append(f"  Test:   {stats['splits']['test']['samples']:6d} samples "
                    f"({stats['splits']['test']['ratio']:.1%}) - {stats['splits']['test']['classes']}")
        if stats['splits']['test']['includes_oneshot']:
            lines.append(f"    └─ includes {stats['splits']['test']['oneshot_samples']} one-shot samples")
        lines.append("="*70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representación concisa para debugging."""
        return (f"CarDataset(regular_classes={len(self.regular_classes)}, "
                f"oneshot_classes={len(self.oneshot_classes)}, "
                f"views={len(self.views)})")


class MPerClassSamplerWrapper:
    """
    Wrapper para MPerClassSampler de pytorch-metric-learning.
    
    Adaptado para trabajar con la estructura de samples de CarDataset.
    El sampler original requiere labels como lista, este wrapper los extrae
    de la estructura de samples.
    """
    
    def __init__(
        self,
        samples: List[Tuple],
        m: int = DEFAULT_K,
        batch_size: int = None,
        length_before_new_iter: int = None
    ):
        """
        Args:
            samples: Lista de samples con estructura (class_key, image_paths, [text_desc]).
            m: Número de muestras por clase por batch (equivalente a K).
            batch_size: Tamaño total del batch (si None, se calcula como m * num_classes_per_batch).
            length_before_new_iter: Número de samples antes de nueva iteración.
        """
        # Extraer labels de samples
        self.samples = samples
        self.labels = []
        self.class_key_to_label = {}
        
        current_label = 0
        for sample in samples:
            class_key = sample[0]
            if class_key not in self.class_key_to_label:
                self.class_key_to_label[class_key] = current_label
                current_label += 1
            self.labels.append(self.class_key_to_label[class_key])
        
        # Calcular batch_size si no se proporciona
        num_classes = len(set(self.labels))
        if batch_size is None:
            # batch_size por defecto: 4 clases × m samples
            batch_size = min(4, num_classes) * m
        
        # Calcular length_before_new_iter si no se proporciona
        if length_before_new_iter is None:
            length_before_new_iter = len(self.labels)
        
        # Crear MPerClassSampler de pytorch-metric-learning
        self.sampler = MPerClassSampler(
            labels=self.labels,
            m=m,
            batch_size=batch_size,
            length_before_new_iter=length_before_new_iter
        )
        
        logging.info(f"MPerClassSamplerWrapper: {len(set(self.labels))} clases, "
                    f"{len(self.labels)} muestras, m={m}, batch_size={batch_size}")
    
    def __iter__(self):
        """Retorna iterador que produce índices individuales."""
        return iter(self.sampler)
    
    def __len__(self):
        return len(self.sampler)


def robust_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Función de collate robusta que maneja errores."""
    valid_batch = [sample for sample in batch if sample is not None]
    
    if len(valid_batch) == 0:
        raise RuntimeError("Todos los samples del batch fallaron")
    
    if len(valid_batch) < len(batch):
        logging.warning(f"Descartados {len(batch) - len(valid_batch)} samples inválidos del batch")

    return default_collate(valid_batch)


def create_car_dataset(
    df: pd.DataFrame,
    views: List[str] = DEFAULT_VIEWS,
    min_images: int = 6,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    train_transform=None,
    val_transform=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    sampling_strategy: str = 'standard',
    P: int = DEFAULT_P,
    K: int = DEFAULT_K,
    **dataset_kwargs
) -> Dict[str, Any]:
    """
    Crea dataset con splits por porcentaje y DataLoaders.
    
    Args:
        df: DataFrame con datos del dataset.
        views: Lista de vistas a incluir.
        min_images: Mínimo de imágenes para clases regulares (default: 6).
        train_ratio: Proporción para train (default: 0.7).
        val_ratio: Proporción para val (default: 0.2).
        test_ratio: Proporción para test (default: 0.1).
        train_transform: Transformaciones para train.
        val_transform: Transformaciones para val/test.
        batch_size: Tamaño de batch.
        num_workers: Número de workers.
        seed: Semilla.
        sampling_strategy: 'standard' o 'pk'.
            - 'standard': Shuffle estándar
            - 'pk': P clases × K muestras por batch (para metric learning)
        P: Clases por batch (solo para 'pk').
        K: Muestras por clase (solo para 'pk').
        **dataset_kwargs: Argumentos adicionales para CarDataset.
    
    Returns:
        Diccionario con 'dataset', 'train_loader', 'val_loader', 'test_loader', 'train_sampler'.
    """
    # Validar estrategia
    valid_strategies = ['standard', 'pk']
    if sampling_strategy not in valid_strategies:
        raise ValueError(f"sampling_strategy debe ser uno de {valid_strategies}")
    
    # Crear dataset base
    base_dataset = CarDataset(
        df=df,
        views=views,
        min_images=min_images,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        transform=None,
        seed=seed,
        **dataset_kwargs
    )

    # Crear transforms si no se proporcionaron
    if train_transform is None:
        train_transform = create_standard_transform(augment=True)
    if val_transform is None:
        val_transform = create_standard_transform(augment=False)

    # Train dataset y loader
    train_dataset = copy.deepcopy(base_dataset)
    train_dataset.transform = train_transform
    train_dataset.set_split('train')

    # Crear sampler según estrategia
    if sampling_strategy == 'standard':
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        effective_batch_size = batch_size
        logging.info(f"Train loader: Standard shuffle, batch_size={effective_batch_size}")
    
    elif sampling_strategy == 'pk':
        # Usar MPerClassSampler de pytorch-metric-learning
        train_sampler = MPerClassSamplerWrapper(
            base_dataset.train_samples,
            m=K,  # K muestras por clase
            batch_size=P * K,  # P clases × K muestras
            length_before_new_iter=len(base_dataset.train_samples)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=P * K,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        effective_batch_size = P * K
        logging.info(f"Train loader: P×K sampling (MPerClassSampler), P={P}, K={K}, batch_size={effective_batch_size}")
    
    # Val loader
    val_dataset = copy.deepcopy(base_dataset)
    val_dataset.transform = val_transform
    val_dataset.set_split('val')
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=robust_collate_fn
    )
    
    # Test loader
    test_dataset = copy.deepcopy(base_dataset)
    test_dataset.transform = val_transform
    test_dataset.set_split('test')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=robust_collate_fn
    )
    
    logging.info("Transforms configurados:")
    logging.info("  - Train: CON augmentación")
    logging.info("  - Val/Test: SIN augmentación")
    
    return {
        "dataset": base_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler
    }


def validate_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Valida la estructura de un DataFrame."""
    required_columns = {'model', 'released_year', 'viewpoint', 'image_path', 'make'}
    missing_columns = required_columns - set(df.columns)
    
    validation = {
        "valid": len(missing_columns) == 0,
        "missing_columns": list(missing_columns),
        "total_records": len(df),
        "unique_models": df['model'].nunique() if 'model' in df.columns else 0,
        "unique_years": df['released_year'].nunique() if 'released_year' in df.columns else 0,
        "unique_viewpoints": df['viewpoint'].nunique() if 'viewpoint' in df.columns else 0,
        "available_viewpoints": df['viewpoint'].unique().tolist() if 'viewpoint' in df.columns else [],
        "unique_makes": df['make'].nunique() if 'make' in df.columns else 0
    }

    if validation["valid"]:
        logging.info("✓ DataFrame estructura válida")
        logging.info(f"  Total records: {validation['total_records']}")
        logging.info(f"  Models: {validation['unique_models']}")
        logging.info(f"  Makes: {validation['unique_makes']}")
        logging.info(f"  Years: {validation['unique_years']}")
        logging.info(f"  Viewpoints: {validation['available_viewpoints']}")
    else:
        logging.error(f"✗ DataFrame inválido. Columnas faltantes: {validation['missing_columns']}")

    return validation
