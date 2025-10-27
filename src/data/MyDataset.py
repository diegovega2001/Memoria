"""
Módulo de dataset personalizado para clasificación de vehículos multi-vista con estrategia de mínimos fijos.

Este módulo proporciona clases para manejar datasets de vehículos con múltiples
vistas, incluyendo funcionalidades de división train/val/test con mínimos configurables,
class-balanced sampling, augmentación y generación de descripciones textuales.
"""

from __future__ import annotations

import copy
import logging
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, Sampler, DataLoader, default_collate


from src.defaults import (
    DEFAULT_VIEWS, DEFAULT_SEED, DEFAULT_P, DEFAULT_K, DEFAULT_MODEL_TYPE,
    DEFAULT_DESCRIPTION_INCLUDE, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, 
    MODEL_TYPES, DESCRIPTION_OPTIONS, ERROR_INVALID_DESCRIPTION, ERROR_INVALID_MODEL_TYPE,
    CLASS_GRANULARITY_OPTIONS, DEFAULT_CLASS_GRANULARITY
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
    Dataset personalizado para clasificación de vehículos multi-vista con mínimos fijos.

    Esta clase maneja datasets de vehículos con múltiples puntos de vista,
    proporcionando funcionalidades de división con mínimos configurables por split,
    augmentación y generación de descripciones textuales para modelos multimodales.
    
    Soporta dos estrategias de granularidad de clase:
    - 'model': Clase = modelo (agrupa años) 
    - 'model+year': Clase = modelo + año 
    
    Soporta zero-shot testing: clases que no aparecen en train/val, útil para evaluar 
    generalización. Prioriza clases temporales (mismo modelo, distinto año) e intra-marca
    (mismo fabricante, distinto modelo).

    Attributes:
        df: DataFrame con los datos del dataset.
        views: Lista de vistas/viewpoints a incluir.
        num_views: Número de vistas configuradas.
        class_granularity: 'model' o 'model+year'.
        min_train_images: Mínimo de imágenes para train.
        min_val_images: Mínimo de imágenes para val.
        min_test_images: Mínimo de imágenes para test.
        seed: Semilla para reproducibilidad.
        transform: Transformaciones a aplicar a las imágenes.
        model_type: Tipo de modelo ('vision', 'textual', 'both').
        description_include: Información para descripciones textuales en train/val:
            - 'type': make + type + model (sin año)
            - '' | 'all' | 'released_year': make + type + model + año (completo)
            Nota: test y zero-shot SIEMPRE usan: make + type (sin model ni año)
        enable_zero_shot: Si incluir clases zero-shot en test.
        num_zero_shot_classes: Número de clases zero-shot a incluir.
        seen_classes: Lista de clases usadas en train/val/test.
        zero_shot_classes: Lista de clases para zero-shot testing.
        label_encoder: Encoder para las etiquetas de clases seen.
        train_samples: Muestras de entrenamiento (descripciones según description_include).
        val_samples: Muestras de validación (descripciones según description_include).
        test_samples: Muestras de prueba seen (descripciones incompletas: make + type).
        zero_shot_samples: Muestras de prueba zero-shot (descripciones incompletas: make + type).
        current_split: Split actual ('train', 'val', 'test', 'zero_shot').
    """

    def __init__(
        self,
        df: pd.DataFrame,
        views: List[str] = DEFAULT_VIEWS,
        class_granularity: str = DEFAULT_CLASS_GRANULARITY,
        min_train_images: int = 5,
        min_val_images: int = 3,
        min_test_images: int = 3,
        seed: int = DEFAULT_SEED,
        transform: Optional[Any] = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        description_include: str = DEFAULT_DESCRIPTION_INCLUDE,
        enable_zero_shot: bool = False,
        num_zero_shot_classes: Optional[int] = None
    ) -> None:
        """
        Inicializa el dataset de vehículos con mínimos fijos.

        Args:
            df: DataFrame con columnas requeridas: 'model', 'released_year', 'viewpoint', 'image_path', 'make'.
            views: Lista de viewpoints a incluir. Si None, usa DEFAULT_VIEWS.
            class_granularity: 'model' (agrupa años) o 'model+year' (separa años).
            min_train_images: Mínimo de imágenes por clase para train (default: 5).
            min_val_images: Mínimo de imágenes por clase para val (default: 3).
            min_test_images: Mínimo de imágenes por clase para test (default: 3).
            seed: Semilla para reproducibilidad aleatoria.
            transform: Transformaciones de torchvision para las imágenes.
            model_type: Tipo de salida ('vision', 'textual', 'both').
            description_include: Descripciones en train/val:
                - 'type': make + type + model (sin año)
                - '' | 'all' | 'released_year': make + type + model + año (completo)
                Nota: test/zero-shot siempre usan: make + type (sin model ni año)
            enable_zero_shot: Si incluir clases zero-shot en test.
            num_zero_shot_classes: Número de clases zero-shot a incluir. Si None, usa todas las disponibles.

        Raises:
            CarDatasetError: Si hay errores de configuración o datos.
            ValueError: Si los parámetros son inválidos.
        """
        # Validación de parámetros
        self._validate_parameters(
            df, views, class_granularity, min_train_images, min_val_images,
            min_test_images, model_type, description_include
        )

        # Configuración básica
        self.df = df.copy()
        self.views = views if views is not None else DEFAULT_VIEWS.copy()
        self.num_views = len(self.views)
        self.seed = seed
        self.class_granularity = class_granularity
        self.min_train_images = min_train_images
        self.min_val_images = min_val_images
        self.min_test_images = min_test_images
        self.min_total_images = min_train_images + min_val_images + min_test_images
        self.transform = transform
        self.model_type = model_type
        self.description_include = description_include
        self.current_split = 'train'
        
        # Configuración zero-shot
        self.enable_zero_shot = enable_zero_shot
        self.num_zero_shot_classes = num_zero_shot_classes

        # Configurar random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        logging.info(f"Inicializando CarDataset con {len(self.df)} registros")
        logging.info(f"Granularidad de clase: {self.class_granularity}")
        logging.info(f"Vistas configuradas: {self.views}")
        logging.info(f"Mínimos por split: train={min_train_images}, val={min_val_images}, test={min_test_images}")
        logging.info(f"Zero-shot testing: {'Habilitado' if self.enable_zero_shot else 'Deshabilitado'}")

        # Inicialización de componentes
        self._initialize_classes()
        self.num_classes = len(self.seen_classes)
        self.label_encoder = self._initialize_label_encoder()
        self.df = self._filter_dataframe()

        # Creación de splits
        self._create_data_splits()
        
        log_msg = f"Samples - Train: {len(self.train_samples)}, Val: {len(self.val_samples)}, Test Seen: {len(self.test_samples)}"
        if self.enable_zero_shot:
            log_msg += f", Zero-Shot: {len(self.zero_shot_samples)}"
        logging.info(log_msg)

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        views: Optional[List[str]],
        class_granularity: str,
        min_train_images: int,
        min_val_images: int,
        min_test_images: int,
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
        if min_train_images < 1:
            raise ValueError("min_train_images debe ser ≥1")
        if min_val_images < 1:
            raise ValueError("min_val_images debe ser ≥1")
        if min_test_images < 1:
            raise ValueError("min_test_images debe ser ≥1")
        
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

    def _initialize_classes(self) -> None:
        """
        Identifica clases seen (para train/val/test) y zero-shot según mínimos.
        
        Clases seen: tienen ≥ min_total_images
        Clases zero-shot: tienen ≥1 imagen pero < min_total_images, priorizando:
            1. Temporal: mismo modelo, distinto año
            2. Intra-marca: mismo fabricante, distinto modelo
        """
        logging.info(f"Identificando clases con granularidad '{self.class_granularity}'...")

        # Determinar columnas de agrupación según granularidad
        if self.class_granularity == 'model+year':
            group_cols = ["model", "released_year", "viewpoint"]
        else:  # 'model'
            group_cols = ["model", "viewpoint"]
        
        # Contar imágenes por clase y vista
        counts = self.df.groupby(group_cols).size().unstack(fill_value=0)
        available_views = [v for v in self.views if v in counts.columns]

        if not available_views:
            raise CarDatasetError(f"Ninguna de las vistas especificadas existe en el dataset: {self.views}")
        
        # Total de imágenes por clase
        total_counts = counts[available_views].sum(axis=1)

        # Filtrar clases que tienen al menos 1 imagen en todas las vistas
        viable_candidates = []
        for class_key in total_counts.index:
            if (counts.loc[class_key][self.views] >= 1).all():
                viable_candidates.append(class_key)
        
        logging.info(f"Clases con ≥1 imagen en todas las vistas: {len(viable_candidates)}")

        # Clasificar en seen vs zero-shot
        self.seen_classes = []
        zero_shot_candidates = []
        
        for class_key in viable_candidates:
            total = total_counts.loc[class_key]
            if total >= self.min_total_images:
                self.seen_classes.append(class_key)
            elif self.enable_zero_shot and total >= 1:
                zero_shot_candidates.append(class_key)
        
        logging.info(f"Clases seen (≥{self.min_total_images} imgs): {len(self.seen_classes)}")
        logging.info(f"Candidatas zero-shot disponibles: {len(zero_shot_candidates)}")

        if not self.seen_classes:
            raise CarDatasetError(
                f"No se encontraron clases con ≥{self.min_total_images} imágenes. "
                f"Reduce los mínimos (train={self.min_train_images}, val={self.min_val_images}, test={self.min_test_images})."
            )
        
        # Seleccionar clases zero-shot si está habilitado
        if self.enable_zero_shot and zero_shot_candidates:
            self.zero_shot_classes = self._select_zero_shot_classes(zero_shot_candidates)
        else:
            self.zero_shot_classes = []
        
        logging.info(f"Clases zero-shot seleccionadas: {len(self.zero_shot_classes)}")

    def _select_zero_shot_classes(self, candidates: List) -> List:
        """
        Selecciona clases zero-shot priorizando temporal e intra-marca.
        
        Estrategia:
        1. Temporal: mismo modelo, distinto año
        2. Intra-marca: mismo fabricante, distinto modelo
        3. Inter-marca: fabricante nuevo
        
        Args:
            candidates: Lista de clases candidatas para zero-shot.
            
        Returns:
            Lista de clases seleccionadas para zero-shot.
        """
        if self.class_granularity != 'model+year':
            # Para granularidad 'model', solo podemos hacer intra/inter-marca
            logging.info("Granularidad 'model': solo disponible selección intra/inter-marca")
            return self._select_zero_shot_by_make(candidates)
        
        # Extraer modelos y marcas de clases seen
        seen_models = set()
        seen_makes = set()
        
        for class_key in self.seen_classes:
            model, year = class_key
            seen_models.add(model)
            
            # Obtener marca del modelo
            model_rows = self.df[self.df['model'] == model]
            if not model_rows.empty:
                make = model_rows.iloc[0]['make']
                seen_makes.add(make)
        
        logging.info(f"Seen: {len(seen_models)} modelos, {len(seen_makes)} marcas")
        
        # Clasificar candidatas
        temporal = []  # Mismo modelo, distinto año
        intra_make = []  # Distinto modelo, misma marca
        inter_make = []  # Marca nueva
        
        for class_key in candidates:
            model, year = class_key
            
            # Obtener marca
            model_rows = self.df[self.df['model'] == model]
            if model_rows.empty:
                continue
            make = model_rows.iloc[0]['make']
            
            # Clasificar
            if model in seen_models:
                temporal.append(class_key)
            elif make in seen_makes:
                intra_make.append(class_key)
            else:
                inter_make.append(class_key)
        
        logging.info(f"Candidatas zero-shot - Temporal: {len(temporal)}, Intra-marca: {len(intra_make)}, Inter-marca: {len(inter_make)}")
        
        # Determinar cuántas clases seleccionar
        if self.num_zero_shot_classes is None:
            # Usar todas las disponibles (prioridad temporal > intra > inter)
            n_total = len(candidates)
        else:
            n_total = min(self.num_zero_shot_classes, len(candidates))
        
        # Seleccionar con prioridad
        selected = []
        
        # Primero todas las temporales disponibles
        selected.extend(temporal)
        
        # Luego intra-marca hasta completar
        if len(selected) < n_total:
            n_needed = n_total - len(selected)
            selected.extend(intra_make[:n_needed])
        
        # Finalmente inter-marca si aún falta
        if len(selected) < n_total:
            n_needed = n_total - len(selected)
            selected.extend(inter_make[:n_needed])
        
        # Shuffle para no tener orden predecible
        random.shuffle(selected)
        
        # Contar distribución final
        final_temporal = len([c for c in selected if c in temporal])
        final_intra = len([c for c in selected if c in intra_make])
        final_inter = len([c for c in selected if c in inter_make])
        
        logging.info(f"Zero-shot seleccionadas: {final_temporal} temporal, {final_intra} intra-marca, {final_inter} inter-marca")
        
        return selected
    
    def _select_zero_shot_by_make(self, candidates: List) -> List:
        """
        Selecciona clases zero-shot solo por marca (para granularidad 'model').
        
        Args:
            candidates: Lista de clases candidatas.
            
        Returns:
            Lista de clases seleccionadas.
        """
        # Obtener marcas seen
        seen_makes = set()
        for class_key in self.seen_classes:
            model = class_key if isinstance(class_key, str) else class_key[0]
            model_rows = self.df[self.df['model'] == model]
            if not model_rows.empty:
                seen_makes.add(model_rows.iloc[0]['make'])
        
        # Clasificar candidatas
        intra_make = []  # Misma marca
        inter_make = []  # Marca nueva
        
        for class_key in candidates:
            model = class_key if isinstance(class_key, str) else class_key[0]
            model_rows = self.df[self.df['model'] == model]
            if model_rows.empty:
                continue
            make = model_rows.iloc[0]['make']
            
            if make in seen_makes:
                intra_make.append(class_key)
            else:
                inter_make.append(class_key)
        
        logging.info(f"Candidatas zero-shot - Intra-marca: {len(intra_make)}, Inter-marca: {len(inter_make)}")
        
        # Seleccionar
        if self.num_zero_shot_classes is None:
            n_total = len(candidates)
        else:
            n_total = min(self.num_zero_shot_classes, len(candidates))
        
        selected = []
        selected.extend(intra_make[:n_total])
        
        if len(selected) < n_total:
            n_needed = n_total - len(selected)
            selected.extend(inter_make[:n_needed])
        
        random.shuffle(selected)
        
        final_intra = len([c for c in selected if c in intra_make])
        final_inter = len([c for c in selected if c in inter_make])
        logging.info(f"Zero-shot seleccionadas: {final_intra} intra-marca, {final_inter} inter-marca")
        
        return selected

    def _initialize_label_encoder(self) -> LabelEncoder:
        """Inicializa el encoder de etiquetas para las clases seen."""
        label_encoder = LabelEncoder()

        # Crear strings de etiquetas según granularidad
        if self.class_granularity == 'model+year':
            class_strings = [f"{model}_{year}" for model, year in self.seen_classes]
        else: 
            class_strings = [str(model) for model in self.seen_classes]

        # Fitting del encoder
        label_encoder.fit(class_strings)

        logging.info(f"LabelEncoder inicializado con {len(self.seen_classes)} clases seen")

        return label_encoder

    def _filter_dataframe(self) -> pd.DataFrame:
        """Filtra el DataFrame para incluir solo clases y vistas válidas (seen + zero-shot)."""
        initial_size = len(self.df)

        # Combinar clases seen y zero-shot
        all_valid_classes = set(self.seen_classes + self.zero_shot_classes)
        
        df_filtered = self.df.copy()
        
        if self.class_granularity == 'model+year':
            df_filtered['class_key'] = list(zip(df_filtered['model'], df_filtered['released_year']))
        else:  # 'model'
            df_filtered['class_key'] = df_filtered['model']

        # Filtrado
        filtered_df = df_filtered[
            (df_filtered['class_key'].isin(all_valid_classes)) &
            (df_filtered['viewpoint'].isin(self.views))
        ].drop('class_key', axis=1)

        final_size = len(filtered_df)
        logging.info(f"DataFrame filtrado: {initial_size} → {final_size} registros")

        return filtered_df

    def _create_text_descriptor(self, image_paths: Union[str, List[str]], incomplete: bool = False) -> str:
        """
        Crea descriptor textual para una imagen o par de imágenes.

        Args:
            image_paths: Ruta(s) de imagen(es) para describir.
            incomplete: Si True, genera descripción incompleta (solo make y type, sin model ni year).
                       Útil para test/zero-shot donde no se debe revelar la clase.

        Returns:
            Descripción textual del vehículo.

        Raises:
            CarDatasetError: Si no se encuentran las imágenes en el dataset.
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
        type_str = rows[0]['type'].strip()
        model = rows[0]['model'].strip()
        year = rows[0]['released_year'].strip()
        viewpoints = [row['viewpoint'] for row in rows]

        # Construir descripción base con viewpoints
        if len(viewpoints) == 1:
            view_prefix = f"The {viewpoints[0]} view image of a"
        else:
            viewpoint_text = " and ".join(viewpoints)
            view_prefix = f"The {viewpoint_text} view images of a"
        
        # Caso 1: Descripción incompleta (solo make + type) - para test/zero-shot
        if incomplete or self.description_include == 'basic':
            desc = f"{view_prefix} {make} vehicle type {type_str}."
        
        # Caso 2: make + type + model (sin año)
        elif self.description_include == 'model':
            desc = f"{view_prefix} {make} {model} vehicle type {type_str}."
        
        # Caso 3: make + type + model + año (completo)
        else:  # self.description_include in ['', 'all', 'released_year']
            desc = f"{view_prefix} {make} {model} vehicle from {year} type {type_str}."
        
        return desc

    def _create_data_splits(self) -> None:
        """Crea las divisiones train/val/test (seen) y zero-shot con mínimos fijos."""
        logging.info(f"Creando splits con mínimos: train={self.min_train_images}, val={self.min_val_images}, test={self.min_test_images}...")

        train_samples, val_samples, test_samples = [], [], []

        # Procesar clases seen
        for class_key in self.seen_classes:
            t, v, te = self._split_class(class_key)
            train_samples.extend(t)
            val_samples.extend(v)
            test_samples.extend(te)

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        
        # Crear zero-shot samples si está habilitado
        if self.enable_zero_shot and self.zero_shot_classes:
            self.zero_shot_samples = self._create_zero_shot_samples()
        else:
            self.zero_shot_samples = []

    def _split_class(self, class_key: Union[Tuple[str, Any], str]) -> Tuple[List, List, List]:
        """
        Divide una clase en train/val/test respetando mínimos fijos.
        
        Estrategia:
        - Asigna min_train_images a train
        - Asigna min_val_images a val
        - Asigna min_test_images a test
        """
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

        # Agrupar por vista y obtener paths
        grouped = class_data.groupby('viewpoint')
        view_images = {}
        for view in self.views:
            if view in grouped.groups:
                paths = list(grouped.get_group(view)['image_path'])
                random.shuffle(paths)
                view_images[view] = paths
            else:
                view_images[view] = []

        # Calcular mínimo de imágenes entre vistas
        min_images = min(len(view_images[view]) for view in self.views)
        
        # Asignar según mínimos fijos
        n_train = self.min_train_images
        n_val = self.min_val_images
        n_test = self.min_test_images
        
        # El resto va a train
        remaining = min_images - (n_train + n_val + n_test)
        if remaining > 0:
            n_train += remaining
        
        return self._create_samples_for_splits(view_images, class_key, n_train, n_val, n_test, min_images)

    def _create_samples_for_splits(
        self, 
        view_images: Dict[str, List[str]], 
        class_key: Union[Tuple[str, Any], str],
        n_train: int, 
        n_val: int,
        n_test: int,
        min_images: int
    ) -> Tuple[List, List, List]:
        """Crea muestras para train/val/test con descripciones apropiadas."""
        train_samples = []
        val_samples = []
        test_samples = []

        for i in range(min_images):
            image_pair = [view_images[view][i] for view in self.views if i < len(view_images[view])]
            if len(image_pair) == len(self.views):  # Solo si hay imagen para cada vista
                # Determinar si es train, val o test
                if i < n_train:
                    # Train: descripción completa
                    if self.model_type in ['textual', 'both']:
                        text_desc = self._create_text_descriptor(image_pair, incomplete=False)
                        sample = (class_key, image_pair, text_desc)
                    else:
                        sample = (class_key, image_pair)
                    train_samples.append(sample)
                elif i < n_train + n_val:
                    # Val: descripción completa
                    if self.model_type in ['textual', 'both']:
                        text_desc = self._create_text_descriptor(image_pair, incomplete=False)
                        sample = (class_key, image_pair, text_desc)
                    else:
                        sample = (class_key, image_pair)
                    val_samples.append(sample)
                elif i < n_train + n_val + n_test:
                    # Test: descripción incompleta (solo make y type)
                    if self.model_type in ['textual', 'both']:
                        text_desc = self._create_text_descriptor(image_pair, incomplete=True)
                        sample = (class_key, image_pair, text_desc)
                    else:
                        sample = (class_key, image_pair)
                    test_samples.append(sample)
                else:
                    break  
                    
        return train_samples, val_samples, test_samples

    def _create_zero_shot_samples(self) -> List:
        """Crea muestras para clases zero-shot."""
        logging.info(f"Creando samples zero-shot para {len(self.zero_shot_classes)} clases...")
        
        zero_shot_samples = []
        for class_key in self.zero_shot_classes:
            samples = self._create_samples_for_class(class_key, max_samples=None)
            zero_shot_samples.extend(samples)
        
        return zero_shot_samples
    
    def _create_samples_for_class(self, class_key: Union[Tuple[str, Any], str], max_samples: Optional[int] = None) -> List:
        """Crea muestras para una clase específica (para zero-shot con descripción incompleta)."""
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
        
        if max_samples is not None:
            min_images = min(min_images, max_samples)
        
        samples = []
        for i in range(min_images):
            image_pair = [view_images[view][i] for view in self.views if i < len(view_images[view])]
            if len(image_pair) == len(self.views):
                if self.model_type in ['textual', 'both']:
                    # Zero-shot: descripción incompleta (solo make y type)
                    text_desc = self._create_text_descriptor(image_pair, incomplete=True)
                    sample = (class_key, image_pair, text_desc)
                else:
                    sample = (class_key, image_pair)
                samples.append(sample)
        
        return samples

    def set_split(self, split: str) -> None:
        """
        Cambia el split actual del dataset.

        Args:
            split: 'train', 'val', 'test', o 'zero_shot'.

        Raises:
            ValueError: Si el split no es válido.
        """
        valid_splits = ['train', 'val', 'test', 'zero_shot']
        if split not in valid_splits:
            raise ValueError(f"split debe ser uno de {valid_splits}")
        
        self.current_split = split
        logging.info(f"Dataset configurado para split: {split}")

    def get_current_samples(self) -> List:
        """Obtiene las muestras del split actual."""
        if self.current_split == 'train':
            return self.train_samples
        elif self.current_split == 'val':
            return self.val_samples
        elif self.current_split == 'test':
            return self.test_samples
        elif self.current_split == 'zero_shot':
            return self.zero_shot_samples
        else:
            raise ValueError(f"Split no válido: {self.current_split}")

    def __len__(self) -> int:
        """Retorna el número de muestras del split actual."""
        return len(self.get_current_samples())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Obtiene una muestra del dataset según el split actual.
        
        Si hay un error al cargar la muestra solicitada, intenta cargar
        una muestra alternativa aleatoria para evitar romper el batch completo.

        Args:
            idx: Índice de la muestra.

        Returns:
            Diccionario con imágenes, etiquetas y descripción textual (si aplica).

        Raises:
            RuntimeError: Si no se puede cargar ninguna muestra después de varios intentos.
        """
        try:
            return self._load_sample(idx)
        except Exception as e:
            logging.warning(
                f"Error cargando muestra {idx} del split {self.current_split}: "
                f"{type(e).__name__}: {e}. Intentando muestra alternativa..."
            )
            # Intentar con muestras alternativas (máximo 3 intentos)
            current_samples = self.get_current_samples()
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    alt_idx = random.randint(0, len(current_samples) - 1)
                    if alt_idx == idx:
                        continue
                    
                    logging.info(f"Intento {retry + 1}/{max_retries}: Cargando muestra alternativa {alt_idx}")
                    return self._load_sample(alt_idx)
                    
                except Exception as retry_error:
                    logging.warning(
                        f"Intento {retry + 1} falló al cargar muestra {alt_idx}: "
                        f"{type(retry_error).__name__}: {retry_error}"
                    )
                    continue
            
            raise RuntimeError(
                f"No se pudo cargar muestra {idx} ni ninguna de {max_retries} alternativas "
                f"del split {self.current_split}"
            )
    
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """
        Carga una muestra específica del dataset.
        
        Args:
            idx: Índice de la muestra a cargar.
            
        Returns:
            Diccionario con la muestra procesada.
            
        Raises:
            Exception: Si hay algún error al cargar la muestra.
        """
        try:
            current_samples = self.get_current_samples()
            sample = current_samples[idx]
            
            # Extraer información de la muestra
            if len(sample) == 3:
                model_year_tuple, paths, text_desc = sample
            else:
                model_year_tuple, paths = sample
                text_desc = None

            # Cargar imágenes con bounding boxes si están disponibles
            images = []
            for path in paths:
                try:
                    img = Image.open(path).convert("RGB")
                    # Obtener bounding box para esta imagen si el transform la soporta
                    bbox = None
                    if hasattr(self.transform, 'use_bbox') and self.transform.use_bbox:
                        matching_rows = self.df[self.df['image_path'] == path]
                        if not matching_rows.empty:
                            bbox = matching_rows.iloc[0].get('bbox')
                    
                    images.append((img, bbox))
                except Exception as e:
                    raise FileNotFoundError(f"No se pudo cargar imagen {path}: {e}") from e

            # Aplicar transformaciones
            if self.transform:
                processed_images = []
                for img_idx, (img, bbox) in enumerate(images):
                    try:
                        if hasattr(self.transform, 'use_bbox') and self.transform.use_bbox and bbox is not None:
                            processed_img = self.transform(img, bbox=bbox)
                        else:
                            processed_img = self.transform(img)

                        processed_images.append(processed_img)
                    except Exception as e:
                        logging.error(
                            f"Error aplicando transformaciones a imagen {img_idx} (path: {paths[img_idx]}): "
                            f"{type(e).__name__}: {e}"
                        )
                        raise RuntimeError(
                            f"Error crítico al transformar imagen {img_idx} del sample {idx}: {e}"
                        ) from e
                images = processed_images
            else:
                images = [img for img, _ in images]
            
            # Crear etiqueta según granularidad (solo para clases seen)
            # Para zero-shot, la etiqueta será -1 (no está en label_encoder)
            if self.class_granularity == 'model+year':
                class_string = f"{model_year_tuple[0]}_{model_year_tuple[1]}"
            else:  # 'model'
                class_string = str(model_year_tuple) if isinstance(model_year_tuple, str) else model_year_tuple[0]
            
            # Verificar si es clase seen o zero-shot
            if model_year_tuple in self.seen_classes:
                label = torch.tensor(
                    self.label_encoder.transform([class_string])[0], 
                    dtype=torch.long
                )
            else:
                # Clase zero-shot: etiqueta -1
                label = torch.tensor(-1, dtype=torch.long)
            
            # Convertir lista de imágenes a tensor
            if len(images) == 1:
                image_tensor = images[0]
            else:
                image_tensor = torch.stack(images)
            
            # Diccionario de salida
            output = {
                "images": image_tensor,
                "labels": label,
                "class_name": class_string
            }
            
            if text_desc is not None:
                output["text_description"] = text_desc

            return output

        except Exception:
            raise

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del dataset."""
        train_stats = self._calculate_split_stats(self.train_samples, "train")
        val_stats = self._calculate_split_stats(self.val_samples, "validation")
        test_stats = self._calculate_split_stats(self.test_samples, "test")
        zero_shot_stats = self._calculate_split_stats(self.zero_shot_samples, "zero_shot") if self.enable_zero_shot else {}
        
        return {
            "overview": {
                "num_seen_classes": self.num_classes,
                "num_zero_shot_classes": len(self.zero_shot_classes),
                "class_granularity": self.class_granularity,
                "num_views": self.num_views,
                "views": self.views,
                "min_train_images": self.min_train_images,
                "min_val_images": self.min_val_images,
                "min_test_images": self.min_test_images,
                "model_type": self.model_type,
                "zero_shot_enabled": self.enable_zero_shot
            },
            "splits": {
                "train": train_stats,
                "validation": val_stats,
                "test": test_stats,
                "zero_shot": zero_shot_stats
            },
            "sample_seen_classes": self.seen_classes[:10],
            "sample_zero_shot_classes": self.zero_shot_classes[:10] if self.zero_shot_classes else []
        }

    def _calculate_split_stats(self, samples: List, split_name: str) -> Dict:
        """Calcula estadísticas para una división del dataset."""
        if len(samples) == 0:
            return {"total_samples": 0}
        
        # Obtener las clases relevantes según el split
        if split_name == "zero_shot":
            relevant_classes = self.zero_shot_classes
        else:
            relevant_classes = self.seen_classes
        
        # Contar samples por clase
        samples_per_class = []
        for class_key in relevant_classes:
            count = len([s for s in samples if s[0] == class_key])
            samples_per_class.append(count)
        
        samples_array = np.array(samples_per_class)
        
        return {
            "total_samples": len(samples),
            "samples_per_class_mean": float(samples_array.mean()) if len(samples_array) > 0 else 0,
            "samples_per_class_std": float(samples_array.std()) if len(samples_array) > 0 else 0,
            "samples_per_class_min": int(samples_array.min()) if len(samples_array) > 0 else 0,
            "samples_per_class_max": int(samples_array.max()) if len(samples_array) > 0 else 0
        }

    def __str__(self) -> str:
        """Representación string detallada del dataset."""
        lines = ["=== Car Dataset Overview (Fixed Minimums Strategy) ==="]
        lines.append(f"Class granularity: {self.class_granularity}")
        lines.append(f"Views: {self.views}")
        lines.append(f"Seen classes: {self.num_classes}")
        lines.append(f"Zero-shot classes: {len(self.zero_shot_classes)}")
        lines.append(f"Model type: {self.model_type}")
        lines.append(f"Description includes: {self.description_include or 'basic info only'}")
        lines.append(f"Current split: {self.current_split}")
        lines.append("")
        lines.append("Split strategy:")
        lines.append(f"  - Train: {self.min_train_images} imgs minimum (+ remainder)")
        lines.append(f"  - Val: {self.min_val_images} imgs minimum (fixed)")
        lines.append(f"  - Test: {self.min_test_images} imgs minimum (fixed)")
        lines.append(f"  - Total minimum: {self.min_total_images} imgs per class")
        lines.append("")
        
        if self.enable_zero_shot:
            lines.append("Zero-shot testing: Enabled")
            lines.append("  - Strategy: Prioritize temporal (same model, diff year) > intra-make > inter-make")
            if self.num_zero_shot_classes:
                lines.append(f"  - Target count: {self.num_zero_shot_classes} classes")
            else:
                lines.append("  - Using all available zero-shot classes")
            lines.append("")
        
        # Estadísticas de divisiones
        splits_data = [
            ("Train", self.train_samples),
            ("Validation", self.val_samples), 
            ("Test (seen)", self.test_samples)
        ]
        
        if self.enable_zero_shot and self.zero_shot_samples:
            splits_data.append(("Zero-shot", self.zero_shot_samples))
        
        for split_name, samples in splits_data:
            if len(samples) > 0:
                stats = self._calculate_split_stats(samples, split_name.lower().replace(" (seen)", "").replace("-", "_"))
                lines.append(f"{split_name} split:")
                lines.append(f"  Total samples: {stats['total_samples']}")
                lines.append(f"  Samples per class - Mean: {stats['samples_per_class_mean']:.1f}, "
                           f"Std: {stats['samples_per_class_std']:.1f}, "
                           f"Min: {stats['samples_per_class_min']}, "
                           f"Max: {stats['samples_per_class_max']}")
            else:
                lines.append(f"{split_name} split: 0 samples")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representación concisa para debugging."""
        return (f"CarDataset(seen_classes={self.num_classes}, zero_shot_classes={len(self.zero_shot_classes)}, "
                f"granularity={self.class_granularity}, views={len(self.views)}, split={self.current_split})")


class ClassBalancedBatchSampler(Sampler):
    """
    Sampler que balancea clases en cada batch para metric learning.
    
    Este sampler crea batches donde cada clase tiene aproximadamente
    la misma probabilidad de aparecer, independientemente de cuántas
    muestras tenga. Esto es crítico para long-tail distributions donde
    clases minoritarias serían subrepresentadas con sampling uniforme.
    
    Estrategia:
    - Samplea uniformemente por clase (no por muestra)
    - Dentro de cada clase, samplea uniformemente sus muestras
    - Garantiza que clases con pocas muestras aparezcan tanto como abundantes
    
    Args:
        samples: Lista de muestras del dataset de entrenamiento.
        batch_size: Tamaño del batch.
        seed: Semilla para reproducibilidad.
        drop_last: Si True, descarta el último batch si es incompleto.
    """
    
    def __init__(
        self, 
        samples: List[Tuple], 
        batch_size: int,
        seed: int = DEFAULT_SEED,
        drop_last: bool = False
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        
        # Agrupar índices por clase
        self.class_to_indices = defaultdict(list)
        for idx, sample in enumerate(samples):
            class_key = sample[0]
            self.class_to_indices[class_key].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        self.num_classes = len(self.classes)
        
        # Calcular número de batches
        # Cada clase debe aparecer aproximadamente el mismo número de veces
        total_samples = len(samples)
        self.num_batches = total_samples // batch_size
        if not drop_last and total_samples % batch_size != 0:
            self.num_batches += 1
        
        logging.info(f"ClassBalancedBatchSampler: {self.num_classes} clases, "
                    f"batch_size={batch_size}, batches={self.num_batches}")
        
        # Log sobre distribución de clases
        samples_per_class = [len(self.class_to_indices[cls]) for cls in self.classes]
        logging.info(f"  Muestras por clase - Min: {min(samples_per_class)}, "
                    f"Max: {max(samples_per_class)}, Mean: {np.mean(samples_per_class):.1f}")
    
    def __iter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Crear iteradores cíclicos para cada clase
        class_iterators = {}
        for cls in self.classes:
            indices = self.class_to_indices[cls].copy()
            random.shuffle(indices)
            class_iterators[cls] = iter(self._cycle_shuffle(indices))
        
        # Generar batches
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Samplear uniformemente por clase
            for _ in range(self.batch_size):
                # Elegir clase aleatoria
                selected_class = random.choice(self.classes)
                # Obtener siguiente índice de esa clase
                idx = next(class_iterators[selected_class])
                batch_indices.append(idx)
            
            yield batch_indices
    
    def _cycle_shuffle(self, indices):
        """Iterador cíclico que re-shuffle cuando se agota."""
        while True:
            for idx in indices:
                yield idx
            # Re-shuffle para la siguiente vuelta
            random.shuffle(indices)
    
    def __len__(self):
        return self.num_batches


class PKBatchSampler(Sampler):
    """
    Sampler P×K para contrastive learning con class-balanced sampling.
    
    Este sampler crea batches con P clases y K muestras por clase,
    pero samplea clases de forma balanceada (no favorece clases abundantes).
    
    Si una clase tiene menos de K muestras, se repiten índices para alcanzar K.
    El dataset debe aplicar augmentación en train para que las repeticiones
    generen variaciones diferentes.
    
    Combina dos estrategias:
    1. Class-balanced sampling: Cada clase tiene igual probabilidad de ser elegida
    2. P×K structure: Útil para triplet/contrastive losses
    
    Args:
        samples: Lista de muestras del dataset de entrenamiento.
        P: Número de clases por batch.
        K: Número de muestras por clase en cada batch.
        seed: Semilla para reproducibilidad.
        use_augmentation_for_fill: Si True, permite usar clases con <K samples
                                   confiando en augmentación.
    """
    
    def __init__(
        self, 
        samples: List[Tuple], 
        P: int = DEFAULT_P, 
        K: int = DEFAULT_K, 
        seed: int = DEFAULT_SEED,
        use_augmentation_for_fill: bool = True
    ):
        self.samples = samples
        self.P = P
        self.K = K
        self.seed = seed
        self.use_augmentation_for_fill = use_augmentation_for_fill
        
        # Agrupar índices por clase
        self.class_to_indices = defaultdict(list)
        for idx, sample in enumerate(samples):
            class_key = sample[0]
            self.class_to_indices[class_key].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.batch_size = self.P * self.K
        
        # Clasificar clases según disponibilidad
        if self.use_augmentation_for_fill:
            # Todas las clases son válidas (se llenarán con repeticiones + augmentación)
            self.valid_classes = [cls for cls in self.classes if len(self.class_to_indices[cls]) >= 1]
            
            classes_needing_aug = [cls for cls in self.valid_classes if len(self.class_to_indices[cls]) < self.K]
            if classes_needing_aug:
                logging.info(f"PKBatchSampler: {len(classes_needing_aug)} clases con <{self.K} muestras "
                           f"usarán augmentación para llenar batch")
        else:
            # Solo clases con ≥K muestras
            self.valid_classes = [cls for cls in self.classes if len(self.class_to_indices[cls]) >= self.K]
        
        if len(self.valid_classes) < self.P:
            logging.warning(f"Solo {len(self.valid_classes)} clases disponibles. "
                          f"Se ajusta P de {self.P} a {len(self.valid_classes)}")
            self.P = len(self.valid_classes)
            self.batch_size = self.P * self.K
        
        if self.P == 0:
            raise ValueError("No hay clases válidas para PKBatchSampler. "
                           "Verifica que train_samples tenga datos.")
        
        # Calcular número de batches
        # Con class-balanced sampling, podemos crear más batches
        min_samples = min(len(self.class_to_indices[cls]) for cls in self.valid_classes)
        # Cada clase puede aparecer múltiples veces
        self.num_batches = max(1, (len(self.valid_classes) * min_samples) // (self.P * self.K))
        
        logging.info(f"PKBatchSampler configurado: P={self.P}, K={self.K}, "
                    f"clases válidas={len(self.valid_classes)}, batches={self.num_batches}")
        
    def __iter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        for batch_num in range(self.num_batches):
            batch_indices = []
            
            # Seleccionar P clases aleatoriamente (class-balanced)
            selected_classes = random.sample(self.valid_classes, self.P)
            
            for cls in selected_classes:
                class_indices = self.class_to_indices[cls].copy()
                n_available = len(class_indices)
                
                if n_available >= self.K:
                    # Suficientes muestras
                    random.shuffle(class_indices)
                    selected = class_indices[:self.K]
                else:
                    # Llenar con repeticiones (augmentación las diferenciará)
                    selected = class_indices.copy()
                    n_needed = self.K - n_available
                    for _ in range(n_needed):
                        selected.append(random.choice(class_indices))
                
                batch_indices.extend(selected)
            
            # Verificar tamaño de batch
            if len(batch_indices) != self.batch_size:
                logging.warning(f"Batch {batch_num}: tamaño incorrecto {len(batch_indices)} != {self.batch_size}")
                while len(batch_indices) < self.batch_size:
                    random_class = random.choice(self.valid_classes)
                    random_idx = random.choice(self.class_to_indices[random_class])
                    batch_indices.append(random_idx)
                batch_indices = batch_indices[:self.batch_size]
                
            yield batch_indices
            
    def __len__(self):
        return self.num_batches


def robust_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Función de collate robusta que maneja errores en muestras individuales.
    
    Args:
        batch: Lista de muestras del dataset.
        
    Returns:
        Diccionario con batch colado.
        
    Raises:
        RuntimeError: Si todas las muestras del batch fallaron.
    """
    valid_batch = [sample for sample in batch if sample is not None]
    
    if len(valid_batch) == 0:
        raise RuntimeError("Todas las muestras del batch fallaron al cargar")
    
    if len(valid_batch) < len(batch):
        logging.warning(
            f"Se descartaron {len(batch) - len(valid_batch)} muestras del batch debido a errores"
        )

    return default_collate(valid_batch)


def create_car_dataset(
    df: pd.DataFrame,
    views: List[str] = DEFAULT_VIEWS,
    min_train_images: int = 5,
    min_val_images: int = 3,
    min_test_images: int = 3,
    train_transform=None,
    val_transform=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    sampling_strategy: str = 'standard',  # 'standard', 'class_balanced', 'pk'
    P: int = DEFAULT_P,
    K: int = DEFAULT_K,
    **dataset_kwargs
) -> Dict[str, Any]:
    """
    Crea dataset con mínimos fijos y DataLoaders con class-balanced sampling.
    
    Args:
        df: DataFrame con datos del dataset.
        views: Lista de vistas a incluir.
        min_train_images: Mínimo de imágenes por clase para train (default: 5).
        min_val_images: Mínimo de imágenes por clase para val (default: 3).
        min_test_images: Mínimo de imágenes por clase para test (default: 3).
        train_transform: Transformaciones para train (CON augmentación).
        val_transform: Transformaciones para val/test (SIN augmentación).
        batch_size: Tamaño de batch (para 'standard' y 'class_balanced').
        num_workers: Número de workers para DataLoaders.
        seed: Semilla para reproducibilidad.
        sampling_strategy: Estrategia de sampling para train:
            - 'standard': Shuffle estándar (bueno para classification)
            - 'class_balanced': Balance de clases en cada batch (bueno para long-tail)
            - 'pk': P×K sampling (bueno para metric learning)
        P: Número de clases por batch (solo para 'pk').
        K: Número de muestras por clase (solo para 'pk').
        **dataset_kwargs: Argumentos adicionales para CarDataset.
    
    Returns:
        Diccionario con 'dataset', 'train_loader', 'val_loader', 'test_loader', 
        'zero_shot_loader' (si aplica), 'train_sampler'.
        
    Example:
        >>> # Classification con shuffle estándar
        >>> dataset_dict = create_car_dataset(
        ...     df=df, 
        ...     sampling_strategy='standard',
        ...     batch_size=256
        ... )
        >>> 
        >>> # Long-tail con class-balanced sampling
        >>> dataset_dict = create_car_dataset(
        ...     df=df,
        ...     sampling_strategy='class_balanced',
        ...     batch_size=256
        ... )
        >>> 
        >>> # Metric learning con P×K sampling
        >>> dataset_dict = create_car_dataset(
        ...     df=df,
        ...     sampling_strategy='pk',
        ...     P=64, K=4
        ... )
        >>> 
        >>> # CLIP con zero-shot testing
        >>> dataset_dict = create_car_dataset(
        ...     df=df,
        ...     class_granularity='model+year',
        ...     enable_zero_shot=True,
        ...     num_zero_shot_classes=400,
        ...     description_include='make_only',
        ...     batch_size=128
        ... )
    """
    # Validar estrategia
    valid_strategies = ['standard', 'class_balanced', 'pk']
    if sampling_strategy not in valid_strategies:
        raise ValueError(f"sampling_strategy debe ser uno de {valid_strategies}")
    
    # Crear dataset base
    base_dataset = CarDataset(
        df=df,
        views=views,
        min_train_images=min_train_images,
        min_val_images=min_val_images,
        min_test_images=min_test_images,
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
    
    elif sampling_strategy == 'class_balanced':
        train_sampler = ClassBalancedBatchSampler(
            base_dataset.train_samples,
            batch_size=batch_size,
            seed=seed,
            drop_last=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        effective_batch_size = batch_size
        logging.info(f"Train loader: Class-balanced sampling, batch_size={effective_batch_size}")
    
    elif sampling_strategy == 'pk':
        train_sampler = PKBatchSampler(
            base_dataset.train_samples,
            P=P,
            K=K,
            seed=seed,
            use_augmentation_for_fill=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        effective_batch_size = P * K
        logging.info(f"Train loader: P×K sampling, P={P}, K={K}, batch_size={effective_batch_size}")
    
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
    
    # Test loader (seen)
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
    
    # Zero-shot loader (si está habilitado)
    zero_shot_loader = None
    if base_dataset.enable_zero_shot and len(base_dataset.zero_shot_samples) > 0:
        zero_shot_dataset = copy.deepcopy(base_dataset)
        zero_shot_dataset.transform = val_transform
        zero_shot_dataset.set_split('zero_shot')
        
        zero_shot_loader = DataLoader(
            zero_shot_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        logging.info(f"Zero-shot loader: {len(zero_shot_dataset)} samples")
    
    # Log sobre augmentación
    logging.info("Transforms configurados:")
    logging.info("  - Train: CON augmentación (crítico para clases con mínimos)")
    logging.info("  - Val/Test: SIN augmentación")
    
    result = {
        "dataset": base_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler
    }
    
    if zero_shot_loader is not None:
        result["zero_shot_loader"] = zero_shot_loader
    
    return result


def validate_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida la estructura de un DataFrame para usar con CarDataset.

    Args:
        df: DataFrame a validar.

    Returns:
        Diccionario con información de validación.
    """
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
        # Estadísticas adicionales si es válido
        if 'model' in df.columns and 'released_year' in df.columns:
            model_year_counts = df.groupby(['model', 'released_year']).size()
            model_counts = df.groupby(['model']).size()
            
            validation.update({
                "unique_model_year_combinations": len(model_year_counts),
                "unique_models_only": len(model_counts),
                "model_year_combinations_with_min_images": {
                    "1+": (model_year_counts >= 1).sum(),
                    "3+": (model_year_counts >= 3).sum(),
                    "5+": (model_year_counts >= 5).sum(),
                    "8+": (model_year_counts >= 8).sum(),
                    "11+": (model_year_counts >= 11).sum(),  
                    "15+": (model_year_counts >= 15).sum()
                },
                "models_with_min_images": {
                    "1+": (model_counts >= 1).sum(),
                    "3+": (model_counts >= 3).sum(),
                    "5+": (model_counts >= 5).sum(),
                    "8+": (model_counts >= 8).sum(),
                    "11+": (model_counts >= 11).sum(),
                    "15+": (model_counts >= 15).sum()
                },
                "viewpoint_distribution": df['viewpoint'].value_counts().to_dict(),
                "make_distribution": df['make'].value_counts().to_dict() if 'make' in df.columns else {}
            })

    return validation
