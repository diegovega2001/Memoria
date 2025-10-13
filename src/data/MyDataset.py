"""
Módulo de dataset personalizado para clasificación de vehículos multi-vista con estrategia adaptativa.

Este módulo proporciona clases para manejar datasets de vehículos con múltiples
vistas, incluyendo funcionalidades de división train/val/test adaptativa según
distribución long-tail, augmentación y generación de descripciones textuales.
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
from torch.utils.data import Dataset, BatchSampler, DataLoader, default_collate


from src.defaults import (DEFAULT_VIEWS, DEFAULT_SEED, DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS, DEFAULT_P, DEFAULT_K, DEFAULT_MODEL_TYPE,
                      DEFAULT_DESCRIPTION_INCLUDE, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, MODEL_TYPES, DESCRIPTION_OPTIONS,
                      UNKNOWN_VALUES, ADAPTIVE_RATIOS, ERROR_INVALID_DESCRIPTION, ERROR_INVALID_MODEL_TYPE,
                      CLASS_GRANULARITY_OPTIONS, DEFAULT_CLASS_GRANULARITY, DEFAULT_TEST_UNSEEN_ENABLED,
                      DEFAULT_TEST_UNSEEN_COUNT, TEST_UNSEEN_STRATEGY_OPTIONS, DEFAULT_TEST_UNSEEN_STRATEGY,
                      DEFAULT_TEMPORAL_RATIO, DEFAULT_INTRA_MAKE_RATIO, DEFAULT_INTER_MAKE_RATIO)


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
    Dataset personalizado para clasificación de vehículos multi-vista con estrategia adaptativa.

    Esta clase maneja datasets de vehículos con múltiples puntos de vista,
    proporcionando funcionalidades de división adaptativa según distribución long-tail,
    augmentación y generación de descripciones textuales para modelos multimodales.
    
    Soporta dos estrategias de granularidad de clase:
    - 'model': Clase = modelo (agrupa años) → Mejor para Vision Models
    - 'model+year': Clase = modelo + año → Mejor para CLIP
    
    Soporta test unseen: clases que no aparecen en train/val, útil para evaluar 
    generalización temporal, intra-marca e inter-marca.

    Attributes:
        df: DataFrame con los datos del dataset.
        views: Lista de vistas/viewpoints a incluir.
        num_views: Número de vistas configuradas.
        class_granularity: 'model' o 'model+year'.
        min_images_for_abundant_class: Umbral para clasificar clases abundantes.
        seed: Semilla para reproducibilidad.
        transform: Transformaciones a aplicar a las imágenes.
        model_type: Tipo de modelo ('vision', 'textual', 'both').
        description_include: Información adicional para descripciones ('', 'type', 'all', 'make_only').
        test_unseen_enabled: Si incluir clases unseen en test.
        test_unseen_count: Número de clases unseen a incluir.
        test_unseen_strategy: 'balanced', 'temporal_only', 'random'.
        class_combinations: Lista de clases válidas (tuplas o strings).
        abundant_classes: Clases con muchas imágenes.
        few_shot_classes: Clases con pocas imágenes.
        single_shot_classes: Clases con muy pocas imágenes.
        label_encoder: Encoder para las etiquetas.
        train_samples: Muestras de entrenamiento.
        val_samples: Muestras de validación.
        test_samples: Muestras de prueba (seen).
        test_unseen_samples: Muestras de prueba (unseen).
        current_split: Split actual ('train', 'val', 'test', 'test_unseen').
    """

    def __init__(
        self,
        df: pd.DataFrame,
        views: List[str] = DEFAULT_VIEWS,
        class_granularity: str = DEFAULT_CLASS_GRANULARITY,
        min_images_for_abundant_class: int = DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS,
        seed: int = DEFAULT_SEED,
        transform: Optional[Any] = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        description_include: str = DEFAULT_DESCRIPTION_INCLUDE,
        test_unseen_enabled: bool = DEFAULT_TEST_UNSEEN_ENABLED,
        test_unseen_count: int = DEFAULT_TEST_UNSEEN_COUNT,
        test_unseen_strategy: str = DEFAULT_TEST_UNSEEN_STRATEGY,
        temporal_ratio: float = DEFAULT_TEMPORAL_RATIO,
        intra_make_ratio: float = DEFAULT_INTRA_MAKE_RATIO,
        inter_make_ratio: float = DEFAULT_INTER_MAKE_RATIO
    ) -> None:
        
        """
        Inicializa el dataset de vehículos con estrategia adaptativa.

        Args:
            df: DataFrame con columnas requeridas: 'model', 'released_year', 'viewpoint', 'image_path', 'make'.
            views: Lista de viewpoints a incluir. Si None, usa DEFAULT_VIEWS.
            class_granularity: 'model' (agrupa años) o 'model+year' (separa años).
            min_images_for_abundant_class: Umbral para clasificar clases como abundantes (≥7 recomendado).
            seed: Semilla para reproducibilidad aleatoria.
            transform: Transformaciones de torchvision para las imágenes.
            model_type: Tipo de salida ('vision', 'textual', 'both').
            description_include: Información adicional en descripciones ('', 'type', 'all', 'make_only').
            test_unseen_enabled: Si incluir clases unseen en test.
            test_unseen_count: Número de clases unseen a incluir en test.
            test_unseen_strategy: 'balanced' (temporal+intra+inter), 'temporal_only', 'random'.
            temporal_ratio: Proporción de clases unseen tipo temporal (mismo modelo, distinto año).
            intra_make_ratio: Proporción de clases unseen tipo intra-marca (distinto modelo, misma marca).
            inter_make_ratio: Proporción de clases unseen tipo inter-marca (marca nunca vista).

        Raises:
            CarDatasetError: Si hay errores de configuración o datos.
            ValueError: Si los parámetros son inválidos.
        """
        # Validación de parámetros
        self._validate_parameters(
            df, views, class_granularity, min_images_for_abundant_class, 
            model_type, description_include, test_unseen_strategy
        )

        # Configuración básica
        self.df = df.copy()
        self.views = views if views is not None else DEFAULT_VIEWS.copy()
        self.num_views = len(self.views)
        self.class_granularity = class_granularity
        self.min_images_for_abundant_class = min_images_for_abundant_class
        self.seed = seed
        self.transform = transform 
        self.model_type = model_type
        self.description_include = description_include
        self.current_split = 'train'
        
        # Configuración test unseen
        self.test_unseen_enabled = test_unseen_enabled
        self.test_unseen_count = test_unseen_count
        self.test_unseen_strategy = test_unseen_strategy
        self.temporal_ratio = temporal_ratio
        self.intra_make_ratio = intra_make_ratio
        self.inter_make_ratio = inter_make_ratio

        # Configurar random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        logging.info(f"Inicializando CarDataset con {len(self.df)} registros")
        logging.info(f"Granularidad de clase: {self.class_granularity}")
        logging.info(f"Vistas configuradas: {self.views}")
        logging.info(f"Test unseen: {'Habilitado' if self.test_unseen_enabled else 'Deshabilitado'}")

        # Inicialización de componentes con estrategia adaptativa
        self._initialize_class_combinations()
        self.num_classes = len(self.class_combinations)
        self.label_encoder = self._initialize_label_encoder()
        self.df = self._filter_dataframe()

        # Creación de splits adaptativos
        self._create_adaptive_data_splits()
        
        log_msg = f"Samples - Train: {len(self.train_samples)}, Val: {len(self.val_samples)}, Test Seen: {len(self.test_samples)}"
        if self.test_unseen_enabled:
            log_msg += f", Test Unseen: {len(self.test_unseen_samples)}"
        logging.info(log_msg)

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        views: Optional[List[str]],
        class_granularity: str,
        min_images_for_abundant_class: int,
        model_type: str,
        description_include: str,
        test_unseen_strategy: str
    ) -> None:
        """Valida los parámetros de entrada."""
        # Validar DataFrame
        required_columns = {'model', 'released_year', 'viewpoint', 'image_path', 'make'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise CarDatasetError(f"Columnas faltantes en DataFrame: {missing}")
        
        # Chequear si el dataframe esta vacio
        if df.empty:
            raise CarDatasetError("El DataFrame no puede estar vacío")
        
        # Validar granularidad de clase
        if class_granularity not in CLASS_GRANULARITY_OPTIONS:
            raise ValueError(f"class_granularity debe ser uno de {CLASS_GRANULARITY_OPTIONS}")
        
        # Validar parámetros numéricos
        if min_images_for_abundant_class < 7:
            logging.warning(f"min_images_for_abundant_class={min_images_for_abundant_class} < 7. "
                          "Se recomienda ≥7 para garantizar mínimos de 2-2-2 en val/test.")
        
        # Validar opciones categóricas
        if model_type not in MODEL_TYPES:
            raise ValueError(ERROR_INVALID_MODEL_TYPE.format(MODEL_TYPES))
        
        # Validar opciones de descripción
        if description_include not in DESCRIPTION_OPTIONS:
            raise ValueError(ERROR_INVALID_DESCRIPTION.format(DESCRIPTION_OPTIONS))
        
        # Validar estrategia test unseen
        if test_unseen_strategy not in TEST_UNSEEN_STRATEGY_OPTIONS:
            raise ValueError(f"test_unseen_strategy debe ser uno de {TEST_UNSEEN_STRATEGY_OPTIONS}")
        
        # Validar vistas
        if views is not None:
            if not views or not isinstance(views, list):
                raise ValueError("views debe ser una lista no vacía")
            # Obtener vistas validas
            available_views = set(df['viewpoint'].unique())
            invalid_views = set(views) - available_views
            if invalid_views:
                raise CarDatasetError(
                    f"Views no disponibles en el dataset: {invalid_views}. "
                    f"Disponibles: {available_views}"
                )

    def _initialize_class_combinations(self) -> None:
        """Identifica clases válidas según granularidad (≥min_images para train/val/test seen)."""
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

        # ESTRATEGIA SIMPLE: Solo clases con ≥min_images para train/val/test
        viable_candidates = total_counts[total_counts >= self.min_images_for_abundant_class].index.tolist()
        
        logging.info(f"Clases con ≥{self.min_images_for_abundant_class} imágenes: {len(viable_candidates)}")

        # Validar que tengan imágenes en todas las vistas requeridas
        self.class_combinations = []
        for class_key in viable_candidates:
            if (counts.loc[class_key][self.views] >= 1).all():
                self.class_combinations.append(class_key)

        logging.info(f"Clases viables (train/val/test seen): {len(self.class_combinations)}")

        if not self.class_combinations:
            raise CarDatasetError(
                f"No se encontraron clases con ≥{self.min_images_for_abundant_class} imágenes. "
                f"Reduce min_images_for_abundant_class."
            )
        
        # Guardar todas las clases disponibles (incluyendo <min_images) para test unseen
        all_candidates = total_counts[total_counts >= 1].index.tolist()
        self.all_available_classes = []
        for class_key in all_candidates:
            if (counts.loc[class_key][self.views] >= 1).all():
                self.all_available_classes.append(class_key)
        
        logging.info(f"Clases totales disponibles (incluyendo <{self.min_images_for_abundant_class}): {len(self.all_available_classes)}")

    def _initialize_label_encoder(self) -> LabelEncoder:
        """Inicializa el encoder de etiquetas para las clases válidas."""
        label_encoder = LabelEncoder()

        # Crear strings de etiquetas según granularidad
        if self.class_granularity == 'model+year':
            class_strings = [f"{model}_{year}" for model, year in self.class_combinations]
        else:  # 'model'
            class_strings = [str(model) for model in self.class_combinations]

        # Fitting del encoder
        label_encoder.fit(class_strings)

        logging.info(f"LabelEncoder inicializado con {len(self.class_combinations)} clases")

        return label_encoder

    def _filter_dataframe(self) -> pd.DataFrame:
        """Filtra el DataFrame para incluir solo clases y vistas válidas."""
        initial_size = len(self.df)

        # Crear filtro según granularidad
        valid_combinations = set(self.class_combinations)
        df_filtered = self.df.copy()
        
        if self.class_granularity == 'model+year':
            df_filtered['class_key'] = list(zip(df_filtered['model'], df_filtered['released_year']))
        else:  # 'model'
            df_filtered['class_key'] = df_filtered['model']

        # Filtrado
        filtered_df = df_filtered[
            (df_filtered['class_key'].isin(valid_combinations)) &
            (df_filtered['viewpoint'].isin(self.views))
        ].drop('class_key', axis=1)

        final_size = len(filtered_df)
        logging.info(f"DataFrame filtrado: {initial_size} → {final_size} registros")

        return filtered_df

    def _create_text_descriptor(self, image_paths: Union[str, List[str]]) -> str:
        """
        Crea descriptor textual para una imagen o par de imágenes.

        Args:
            image_paths: Ruta(s) de imagen(es) para describir.

        Returns:
            Descripción textual del vehículo.

        Raises:
            CarDatasetError: Si no se encuentran las imágenes en el dataset.
        """
        # Generalizar paths como listas
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Obtener información de las imágenes
        try:
            rows = []
            for path in image_paths:
                # Obtención de las filas a partir de los paths de las imágenes
                matching_rows = self.df[self.df['image_path'] == path]

                # Error: No hay filas disponibles
                if matching_rows.empty:
                    raise CarDatasetError(f"Imagen no encontrada en dataset: {path}")
                
                # Añadir las filas obtenidas a la lista de filas
                rows.append(matching_rows.iloc[0])

        # Error
        except Exception as e:
            raise CarDatasetError(f"Error obteniendo información de imagen: {e}") from e
        
        # Información base
        make = rows[0]['make'].strip()
        model = rows[0]['model'].strip()
        year = rows[0]['released_year']
        viewpoints = [row['viewpoint'] for row in rows]

        # Modo make_only: Solo marca (para val/test en CLIP)
        if self.description_include == 'make_only':
            if len(viewpoints) == 1:
                desc = f"The {viewpoints[0]} view image of a {make} vehicle"
            else:
                viewpoint_text = " and ".join(viewpoints)
                desc = f"The {viewpoint_text} view images of a {make} vehicle"
            return desc + "."
        
        # Modo completo: Incluir modelo y año (para train)
        if len(viewpoints) == 1:
            desc = f"The {viewpoints[0]} view image of a {make} {model} vehicle from {year}"
        else:
            viewpoint_text = " and ".join(viewpoints)
            desc = f"The {viewpoint_text} view images of a {make} {model} vehicle from {year}"

        # Agregar información adicional según configuración
        desc = self._add_additional_info(desc, rows[0])
        
        return desc + "."

    def _add_additional_info(self, desc: str, row: pd.Series) -> str:
        """Agrega información adicional a la descripción."""

        # Añadir tipo de vehículo a la descripción
        if self.description_include in ['type', 'all']:
            vehicle_type = row.get('type')
            if pd.notna(vehicle_type) and vehicle_type not in UNKNOWN_VALUES:
                desc += f", type {vehicle_type}"

        return desc

    def _create_adaptive_data_splits(self) -> None:
        """Crea las divisiones train/val/test (seen) con split fijo mínimos 3-2-2."""
        logging.info("Creando splits del dataset (mínimos 3-2-2)...")

        train_samples, val_samples, test_samples = [], [], []

        # Procesar todas las clases viables con el mismo split
        for class_key in self.class_combinations: 
            t, v, te = self._split_class(class_key)
            train_samples.extend(t)
            val_samples.extend(v) 
            test_samples.extend(te)

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        
        # Crear test unseen si está habilitado
        if self.test_unseen_enabled:
            self.test_unseen_samples = self._create_test_unseen()
        else:
            self.test_unseen_samples = []

    def _split_class(self, class_key: Union[Tuple[str, Any], str]) -> Tuple[List, List, List]:
        """Split proporcional con mínimos garantizados: 2 val, 2 test, resto train."""
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
        if min_images < self.min_images_for_abundant_class:
            logging.warning(f"Clase {class_key} tiene {min_images} imágenes (<{self.min_images_for_abundant_class}), saltando...")
            return [], [], []
        
        # Split proporcional con mínimos garantizados
        # Mínimos absolutos
        MIN_VAL = 2
        MIN_TEST = 2
        MIN_TRAIN = min_images - MIN_VAL - MIN_TEST  # Lo que quede
        
        # Aplicar porcentajes de ADAPTIVE_RATIOS solo si mejoran los mínimos
        ratios = ADAPTIVE_RATIOS['abundant']  # 43% train, 29% val, 29% test
        n_train_ratio = int(min_images * ratios['train'])
        n_val_ratio = int(min_images * ratios['val'])
        n_test_ratio = min_images - n_train_ratio - n_val_ratio
        
        # Usar el máximo entre mínimos y ratios
        n_val = max(MIN_VAL, n_val_ratio)
        n_test = max(MIN_TEST, n_test_ratio)
        n_train = min_images - n_val - n_test
        
        # Garantizar que train tenga al menos algo
        if n_train < 1:
            n_train = 1
            n_val = min(MIN_VAL, (min_images - n_train) // 2)
            n_test = min_images - n_train - n_val
        
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
        """Crea muestras para train/val/test."""
        train_samples = []
        val_samples = []
        test_samples = []

        # Generación de los samples
        for i in range(min_images):
            image_pair = [view_images[view][i] for view in self.views if i < len(view_images[view])]
            if len(image_pair) == len(self.views):  # Solo si hay imagen para cada vista
                if self.model_type in ['textual', 'both']:
                    text_desc = self._create_text_descriptor(image_pair)
                    sample = (class_key, image_pair, text_desc)
                else:
                    sample = (class_key, image_pair)
                
                if i < n_train:
                    train_samples.append(sample)
                elif i < n_train + n_val:
                    val_samples.append(sample)
                else:
                    test_samples.append(sample)
                    
        return train_samples, val_samples, test_samples

    def _create_test_unseen(self) -> List:
        """Crea muestras de test unseen con estrategia balanceada."""
        logging.info(f"Creando test unseen con estrategia '{self.test_unseen_strategy}'...")
        
        # Clases seen (ya usadas en train/val/test)
        seen_classes = set(self.class_combinations)
        
        # Candidatas unseen: clases con ≥1 imagen que NO están en seen
        unseen_candidates = [c for c in self.all_available_classes if c not in seen_classes]
        
        if not unseen_candidates:
            logging.warning("No hay clases disponibles para test unseen")
            return []
        
        logging.info(f"Candidatas unseen: {len(unseen_candidates)}")
        
        # Limitar cantidad
        n_unseen = min(self.test_unseen_count, len(unseen_candidates))
        
        if self.test_unseen_strategy == 'random':
            selected_unseen = random.sample(unseen_candidates, n_unseen)
        
        elif self.test_unseen_strategy == 'balanced':
            # Solo para model+year: balancear temporal/intra-marca/inter-marca
            if self.class_granularity != 'model+year':
                logging.warning("Estrategia 'balanced' solo para model+year, usando 'random'")
                selected_unseen = random.sample(unseen_candidates, n_unseen)
            else:
                selected_unseen = self._select_balanced_unseen(unseen_candidates, n_unseen, seen_classes)
        
        elif self.test_unseen_strategy == 'temporal_only':
            # Solo para model+year: mismo modelo, años diferentes
            if self.class_granularity != 'model+year':
                logging.warning("Estrategia 'temporal_only' solo para model+year, usando 'random'")
                selected_unseen = random.sample(unseen_candidates, n_unseen)
            else:
                selected_unseen = self._select_temporal_unseen(unseen_candidates, n_unseen, seen_classes)
        
        else:
            selected_unseen = random.sample(unseen_candidates, n_unseen)
        
        logging.info(f"Seleccionadas {len(selected_unseen)} clases unseen")
        
        # Crear samples para las clases unseen seleccionadas
        test_unseen_samples = []
        for class_key in selected_unseen:
            samples = self._create_samples_for_class(class_key, max_samples=None)
            test_unseen_samples.extend(samples)
        
        return test_unseen_samples
    
    def _select_balanced_unseen(self, candidates: List, n_total: int, seen_classes: set) -> List:
        """Selecciona clases unseen balanceadas: temporal/intra-marca/inter-marca."""
        # Extraer marcas y modelos vistos
        seen_models = set(c[0] for c in seen_classes)
        seen_makes = set()
        for class_key in seen_classes:
            model, year = class_key
            make_rows = self.df[self.df['model'] == model]
            if not make_rows.empty:
                seen_makes.add(make_rows.iloc[0]['make'])
        
        # Clasificar candidatas
        temporal = []  # Mismo modelo, distinto año
        intra_make = []  # Distinto modelo, misma marca
        inter_make = []  # Marca nueva
        
        for class_key in candidates:
            model, year = class_key
            make_rows = self.df[self.df['model'] == model]
            if make_rows.empty:
                continue
            make = make_rows.iloc[0]['make']
            
            if model in seen_models:
                temporal.append(class_key)
            elif make in seen_makes:
                intra_make.append(class_key)
            else:
                inter_make.append(class_key)
        
        # Calcular cantidades según ratios
        n_temporal = int(n_total * self.temporal_ratio)
        n_intra = int(n_total * self.intra_make_ratio)
        n_inter = n_total - n_temporal - n_intra
        
        # Seleccionar de cada categoría
        selected = []
        selected.extend(random.sample(temporal, min(n_temporal, len(temporal))))
        selected.extend(random.sample(intra_make, min(n_intra, len(intra_make))))
        selected.extend(random.sample(inter_make, min(n_inter, len(inter_make))))
        
        # Rellenar si falta
        while len(selected) < n_total and len(selected) < len(candidates):
            remaining = [c for c in candidates if c not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))
        
        logging.info(f"Test unseen balanceado: {len([c for c in selected if c in temporal])} temporal, "
                    f"{len([c for c in selected if c in intra_make])} intra-marca, "
                    f"{len([c for c in selected if c in inter_make])} inter-marca")
        
        return selected
    
    def _select_temporal_unseen(self, candidates: List, n_total: int, seen_classes: set) -> List:
        """Selecciona solo clases temporales: mismo modelo, años diferentes."""
        seen_models = set(c[0] for c in seen_classes)
        temporal = [c for c in candidates if c[0] in seen_models]
        
        n_selected = min(n_total, len(temporal))
        selected = random.sample(temporal, n_selected)
        
        logging.info(f"Test unseen temporal: {len(selected)} clases")
        return selected
    
    def _create_samples_for_class(self, class_key: Union[Tuple[str, Any], str], max_samples: Optional[int] = None) -> List:
        """Crea muestras para una clase específica (para test unseen)."""
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
                    text_desc = self._create_text_descriptor(image_pair)
                    sample = (class_key, image_pair, text_desc)
                else:
                    sample = (class_key, image_pair)
                samples.append(sample)
        
        return samples

    def set_split(self, split: str) -> None:
        """
        Cambia el split actual del dataset.

        Args:
            split: 'train', 'val', 'test', o 'test_unseen'.

        Raises:
            ValueError: Si el split no es válido.
        """
        valid_splits = ['train', 'val', 'test', 'test_unseen']
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
        elif self.current_split == 'test_unseen':
            return self.test_unseen_samples
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
        # Intentar cargar la muestra original
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
                    # Seleccionar índice aleatorio diferente
                    alt_idx = random.randint(0, len(current_samples) - 1)
                    if alt_idx == idx:
                        continue  # Evitar reintentar el mismo índice
                    
                    logging.info(f"Intento {retry + 1}/{max_retries}: Cargando muestra alternativa {alt_idx}")
                    return self._load_sample(alt_idx)
                    
                except Exception as retry_error:
                    logging.warning(
                        f"Intento {retry + 1} falló al cargar muestra {alt_idx}: "
                        f"{type(retry_error).__name__}: {retry_error}"
                    )
                    continue
            
            # Si todos los intentos fallaron, lanzar excepción
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
                        # Buscar la bbox correspondiente en el DataFrame
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
                        # Log del error pero continuar con la siguiente imagen
                        logging.error(
                            f"Error aplicando transformaciones a imagen {img_idx} (path: {paths[img_idx]}): "
                            f"{type(e).__name__}: {e}"
                        )
                        # Re-lanzar la excepción ya que no podemos continuar sin esta imagen
                        raise RuntimeError(
                            f"Error crítico al transformar imagen {img_idx} del sample {idx}: {e}"
                        ) from e
                images = processed_images
            else:
                # Si no hay transformaciones, extraer solo las imágenes
                images = [img for img, _ in images]
            # Crear etiqueta según granularidad
            if self.class_granularity == 'model+year':
                class_string = f"{model_year_tuple[0]}_{model_year_tuple[1]}"
            else:  # 'model'
                class_string = str(model_year_tuple) if isinstance(model_year_tuple, str) else model_year_tuple[0]
            
            label = torch.tensor(
                self.label_encoder.transform([class_string])[0], 
                dtype=torch.long
            )
            
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
            # Añadir descripción textual
            if text_desc is not None:
                output["text_description"] = text_desc

            return output

        except Exception as e:
            # Re-lanzar la excepción para que __getitem__ pueda manejarla con retry
            raise

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del dataset."""
        # Calcular estadísticas por split
        train_stats = self._calculate_split_stats(self.train_samples, "train")
        val_stats = self._calculate_split_stats(self.val_samples, "validation")
        test_stats = self._calculate_split_stats(self.test_samples, "test")

        test_unseen_stats = self._calculate_split_stats(self.test_unseen_samples, "test_unseen") if self.test_unseen_enabled else {}
        
        return {
            "overview": {
                "num_classes": self.num_classes,
                "class_granularity": self.class_granularity,
                "num_views": self.num_views,
                "views": self.views,
                "min_images_per_class": self.min_images_for_abundant_class,
                "model_type": self.model_type,
                "test_unseen_enabled": self.test_unseen_enabled,
                "test_unseen_count": len(self.test_unseen_samples) if self.test_unseen_enabled else 0
            },
            "splits": {
                "train": train_stats,
                "validation": val_stats,
                "test": test_stats,
                "test_unseen": test_unseen_stats
            },
            "sample_classes": self.class_combinations[:10],
            "total_classes": len(self.class_combinations)
        }

    def _calculate_split_stats(self, samples: List, split_name: str) -> Dict:
        """Calcula estadísticas para una división del dataset."""
        # Samples vacíos
        if len(samples) == 0:
            return {"total_samples": 0}
        # Calculo de samples por clase
        samples_per_class = []
        for class_key in self.class_combinations:
            count = len([s for s in samples if s[0] == class_key])
            samples_per_class.append(count)
        samples_array = np.array(samples_per_class)
        
        return {
            "total_samples": len(samples),
            "samples_per_combination_mean": float(samples_array.mean()),
            "samples_per_combination_std": float(samples_array.std()),
            "samples_per_combination_min": int(samples_array.min()),
            "samples_per_combination_max": int(samples_array.max())
        }

    def __str__(self) -> str:
        """Representación string detallada del dataset."""
        lines = ["=== Car Dataset Overview (Unified Strategy) ==="]
        lines.append(f"Class granularity: {self.class_granularity}")
        lines.append(f"Views: {self.views}")
        lines.append(f"Number of classes: {self.num_classes}")
        lines.append(f"Min images per class: {self.min_images_for_abundant_class}")
        lines.append(f"Model type: {self.model_type}")
        lines.append(f"Description includes: {self.description_include or 'basic info only'}")
        lines.append(f"Current split: {self.current_split}")
        lines.append("")
        lines.append("Split strategy:")
        lines.append(f"  - All classes use proportional split (~43%-29%-29%) with minimums (2-2)")
        lines.append(f"  - Example: 7 imgs → 3-2-2, 10 imgs → 5-2-3, 15 imgs → 7-4-4")
        lines.append("")
        
        if self.test_unseen_enabled:
            lines.append(f"Test unseen: Enabled ({self.test_unseen_strategy})")
            lines.append(f"  - Target count: {self.test_unseen_count}")
            if self.test_unseen_strategy == 'balanced':
                lines.append(f"  - Temporal ratio: {self.temporal_ratio:.0%}")
                lines.append(f"  - Intra-make ratio: {self.intra_make_ratio:.0%}")
                lines.append(f"  - Inter-make ratio: {self.inter_make_ratio:.0%}")
            lines.append("")
        
        # Estadísticas de divisiones
        splits_data = [
            ("Train", self.train_samples),
            ("Validation", self.val_samples), 
            ("Test (seen)", self.test_samples)
        ]
        
        if self.test_unseen_enabled:
            splits_data.append(("Test (unseen)", self.test_unseen_samples))
        
        for split_name, samples in splits_data:
            if len(samples) > 0:
                stats = self._calculate_split_stats(samples, split_name.lower())
                lines.append(f"{split_name} split:")
                lines.append(f"  Total samples: {stats['total_samples']}")
                lines.append(f"  Samples per class - Mean: {stats['samples_per_combination_mean']:.1f}, "
                           f"Std: {stats['samples_per_combination_std']:.1f}, "
                           f"Min: {stats['samples_per_combination_min']}, "
                           f"Max: {stats['samples_per_combination_max']}")
            else:
                lines.append(f"{split_name} split: 0 samples")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representación concisa para debugging."""
        return (f"CarDataset(classes={self.num_classes}, granularity={self.class_granularity}, "
                f"views={len(self.views)}, split={self.current_split}, "
                f"test_unseen={'enabled' if self.test_unseen_enabled else 'disabled'})")


class IdentitySampler(BatchSampler):
    """
    BatchSampler P×K para contrastive learning con soporte para augmentación.
    
    Este sampler crea batches con P clases y K muestras por clase.
    Si una clase tiene menos de K muestras, se repiten índices para alcanzar K.
    El dataset debe aplicar augmentación en train para que las repeticiones
    generen variaciones diferentes de la misma imagen.
    
    Estrategia de llenado:
    - Si clase tiene n < K muestras, se usan las n originales + (K-n) repeticiones con augmentación
    - Ejemplo: clase con 3 muestras y K=4 → usa 3 originales + 1 repetición aumentada
    - Ejemplo: clase con 3 muestras y K=8 → usa 3 originales + 5 repeticiones aumentadas
    
    Args:
        samples: Lista de muestras del dataset de entrenamiento.
        P: Número de clases por batch.
        K: Número de muestras por clase en cada batch.
        seed: Semilla para reproducibilidad.
        use_augmentation_for_fill: Si True, permite usar todas las clases (incluso con <K samples)
                                   confiando en que augmentación creará variaciones.
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
        self.class_to_indices = {}
        for idx, sample in enumerate(samples):
            class_key = sample[0]  # Puede ser tupla (model, year) o string (model)
            if class_key not in self.class_to_indices:
                self.class_to_indices[class_key] = []
            self.class_to_indices[class_key].append(idx)

        self.classes = list(self.class_to_indices.keys())
        self.batch_size = self.P * self.K
        
        # Clasificar clases según disponibilidad
        if self.use_augmentation_for_fill:
            # Todas las clases son válidas, incluso con <K muestras
            # (se llenarán con repeticiones + augmentación)
            self.valid_classes = [cls for cls in self.classes if len(self.class_to_indices[cls]) >= 1]
            
            # Contar clases que necesitan augmentación
            classes_needing_aug = [cls for cls in self.valid_classes if len(self.class_to_indices[cls]) < self.K]
            if classes_needing_aug:
                logging.info(f"IdentitySampler: {len(classes_needing_aug)} clases con <{self.K} muestras "
                           f"usarán augmentación para llenar batch")
        else:
            # Solo clases con ≥K muestras
            self.valid_classes = [cls for cls in self.classes if len(self.class_to_indices[cls]) >= self.K]
        
        if len(self.valid_classes) < self.P:
            logging.warning(f"Solo {len(self.valid_classes)} clases disponibles. "
                          f"Se ajusta P de {self.P} a {len(self.valid_classes)}")
            self.P = len(self.valid_classes)
        
        if self.P == 0:
            raise ValueError("No hay clases válidas para IdentitySampler. "
                           "Verifica que train_samples tenga datos.")
        
        # Calcular número de batches posibles
        # Con augmentation, podemos crear más batches porque no estamos limitados por min_samples
        if self.use_augmentation_for_fill:
            # Usar todas las clases válidas varias veces
            # Cada clase puede aparecer múltiples veces en diferentes batches
            min_samples = min(len(self.class_to_indices[cls]) for cls in self.valid_classes)
            # Batches = cuántas veces podemos usar cada clase antes de agotar incluso la más pequeña
            self.num_batches = max(1, (len(self.valid_classes) // self.P) * min_samples)
        else:
            min_samples_per_class = min(len(self.class_to_indices[cls]) for cls in self.valid_classes)
            self.num_batches = max(1, min_samples_per_class // self.K * len(self.valid_classes) // self.P)
        
        logging.info(f"IdentitySampler configurado: P={self.P}, K={self.K}, "
                    f"clases válidas={len(self.valid_classes)}, batches={self.num_batches}, "
                    f"augmentation_fill={'enabled' if self.use_augmentation_for_fill else 'disabled'}")
        
    def __iter__(self):
        random.seed(self.seed)
        
        for batch_num in range(self.num_batches):
            batch_indices = []
            
            # Seleccionar P clases aleatoriamente
            if len(self.valid_classes) >= self.P:
                selected_classes = random.sample(self.valid_classes, self.P)
            else:
                # Si no hay suficientes clases, usar con reemplazo
                selected_classes = random.choices(self.valid_classes, k=self.P)
            
            for cls in selected_classes:
                class_indices = self.class_to_indices[cls].copy()
                n_available = len(class_indices)
                
                if n_available >= self.K:
                    # Caso simple: suficientes muestras
                    random.shuffle(class_indices)
                    selected = class_indices[:self.K]
                else:
                    # Caso con augmentación: repetir índices para llenar K
                    # Primero usar todas las disponibles
                    selected = class_indices.copy()
                    # Luego rellenar con repeticiones (augmentación las diferenciará)
                    n_needed = self.K - n_available
                    for _ in range(n_needed):
                        # Elegir aleatoriamente de las originales para repetir
                        selected.append(random.choice(class_indices))
                
                batch_indices.extend(selected)
            
            # Verificar tamaño de batch
            if len(batch_indices) != self.batch_size:
                logging.warning(f"Batch {batch_num}: tamaño incorrecto {len(batch_indices)} != {self.batch_size}")
                # Rellenar si falta
                while len(batch_indices) < self.batch_size:
                    random_class = random.choice(self.valid_classes)
                    random_idx = random.choice(self.class_to_indices[random_class])
                    batch_indices.append(random_idx)
                # Truncar si sobra
                batch_indices = batch_indices[:self.batch_size]
                
            yield batch_indices
            
    def __len__(self):
        return self.num_batches


def robust_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Función de collate robusta que maneja errores en muestras individuales.
    
    Si alguna muestra del batch es None (por error en __getitem__), 
    se filtra y se continúa con las muestras válidas.
    
    Args:
        batch: Lista de muestras del dataset.
        
    Returns:
        Diccionario con batch colado, o None si no hay muestras válidas.
        
    Raises:
        RuntimeError: Si todas las muestras del batch fallaron.
    """
    # Filtrar muestras None 
    valid_batch = [sample for sample in batch if sample is not None]
    
    if len(valid_batch) == 0:
        raise RuntimeError("Todas las muestras del batch fallaron al cargar")
    
    if len(valid_batch) < len(batch):
        logging.warning(
            f"Se descartaron {len(batch) - len(valid_batch)} muestras del batch debido a errores"
        )

    # Usar el collate default de PyTorch para las muestras válidas
    return default_collate(valid_batch)


# Funciones de conveniencia
def create_car_dataset(
    df: pd.DataFrame,
    views: List[str] = DEFAULT_VIEWS,
    min_images_for_abundant_class: int = DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS,
    P: int = DEFAULT_P,
    K: int = DEFAULT_K,
    train_transform=None,
    val_transform=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    use_identity_sampler: bool = False,
    **dataset_kwargs
) -> Dict[str, Any]:
    """
    Crea dataset con estrategia unificada y DataLoaders listos para usar.
    
    Args:
        df: DataFrame con datos del dataset.
        views: Lista de vistas a incluir.
        min_images_for_abundant_class: Umbral mínimo de imágenes por clase (≥7 recomendado).
        P: Número de clases por batch para contrastive learning (solo train con IdentitySampler).
        K: Número de muestras por clase por batch (solo train con IdentitySampler).
        train_transform: Transformaciones para train (CON augmentación).
                        Si es None, se crea automáticamente con augmentación.
        val_transform: Transformaciones para val/test (SIN augmentación).
                      Si es None, se crea automáticamente sin augmentación.
        batch_size: Tamaño de batch para val y test (también para train si no se usa IdentitySampler).
        num_workers: Número de workers para DataLoaders.
        seed: Semilla para reproducibilidad.
        use_identity_sampler: Si True, usa IdentitySampler P×K para train (metric learning).
                             Si False, usa batch_size estándar con shuffle (classification).
                             NOTA: Con IdentitySampler, train_transform DEBE tener augmentación
                             para clases con <K muestras.
        **dataset_kwargs: Argumentos adicionales para CarDataset (class_granularity, test_unseen_*, etc).
    
    Returns:
        Diccionario con 'dataset', 'train_loader', 'val_loader', 'test_loader', 'test_unseen_loader', 'train_sampler'.
        
    Example:
        >>> # For classification (Vision models con granularidad 'model')
        >>> dataset_dict = create_car_dataset(
        ...     df=df, 
        ...     class_granularity='model',
        ...     use_identity_sampler=False, 
        ...     batch_size=256
        ... )
        >>> # For metric learning (Vision models con granularidad 'model')
        >>> dataset_dict = create_car_dataset(
        ...     df=df, 
        ...     class_granularity='model',
        ...     use_identity_sampler=True, 
        ...     P=64, K=4
        ... )
        >>> # For CLIP (granularidad 'model+year' con test unseen)
        >>> dataset_dict = create_car_dataset(
        ...     df=df,
        ...     class_granularity='model+year',
        ...     test_unseen_enabled=True,
        ...     test_unseen_count=400,
        ...     test_unseen_strategy='balanced',
        ...     description_include='make_only',  # Para val/test
        ...     batch_size=128
        ... )
    """
    # Crear dataset base para obtener samples (sin transform específico aún)
    base_dataset = CarDataset(
        df=df,
        views=views,
        min_images_for_abundant_class=min_images_for_abundant_class,
        transform=None,  # Sin transform en el base
        seed=seed,
        **dataset_kwargs
    )

    # Crear transforms si no se proporcionaron
    if train_transform is None:
        train_transform = create_standard_transform(augment=True)
    if val_transform is None:
        val_transform = create_standard_transform(augment=False)

    # Crear datasets para cada split con transforms específicos
    train_dataset = copy.deepcopy(base_dataset)
    train_dataset.transform = train_transform
    train_dataset.set_split('train')

    # Train loader: condicional según use_identity_sampler
    if use_identity_sampler:
        # Metric learning: usar IdentitySampler P×K con augmentación habilitada
        train_sampler = IdentitySampler(
            base_dataset.train_samples, 
            P=P, 
            K=K, 
            seed=seed,
            use_augmentation_for_fill=True  # Permite clases con <K muestras
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        effective_batch_size = P * K
        logging.info(f"Train loader: IdentitySampler P={P}, K={K}, batch_size={effective_batch_size}")
    else:
        # Classification: usar batch_size estándar con shuffle
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
        logging.info(f"Train loader: Standard sampler, batch_size={effective_batch_size}")
    
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
    
    # Test unseen loader (si está habilitado)
    test_unseen_loader = None
    if base_dataset.test_unseen_enabled and len(base_dataset.test_unseen_samples) > 0:
        test_unseen_dataset = copy.deepcopy(base_dataset)
        test_unseen_dataset.transform = val_transform
        test_unseen_dataset.set_split('test_unseen')
        
        test_unseen_loader = DataLoader(
            test_unseen_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=robust_collate_fn
        )
        logging.info(f"Test unseen loader: {len(test_unseen_dataset)} samples")
    
    # Log sobre augmentación
    train_has_augment = hasattr(train_transform, 'transforms') or 'augment' in str(type(train_transform)).lower()
    val_has_augment = hasattr(val_transform, 'transforms') or 'augment' in str(type(val_transform)).lower()
    logging.info(f"Transforms configurados:")
    logging.info(f"  - Train: {'CON' if train_has_augment else 'SIN'} augmentación")
    logging.info(f"  - Val/Test: SIN augmentación")
    
    result = {
        "dataset": base_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler
    }
    
    if test_unseen_loader is not None:
        result["test_unseen_loader"] = test_unseen_loader
    
    return result


def validate_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida la estructura de un DataFrame para usar con CarDataset.

    Args:
        df: DataFrame a validar.

    Returns:
        Diccionario con información de validación.
    """
    required_columns = {'model', 'released_year', 'viewpoint', 'image_path'}
    missing_columns = required_columns - set(df.columns)
    
    validation = {
        "valid": len(missing_columns) == 0,
        "missing_columns": list(missing_columns),
        "total_records": len(df),
        "unique_models": df['model'].nunique() if 'model' in df.columns else 0,
        "unique_years": df['released_year'].nunique() if 'released_year' in df.columns else 0,
        "unique_viewpoints": df['viewpoint'].nunique() if 'viewpoint' in df.columns else 0,
        "available_viewpoints": df['viewpoint'].unique().tolist() if 'viewpoint' in df.columns else []
    }

    if validation["valid"]:
        # Estadísticas adicionales si es válido
        if 'model' in df.columns and 'released_year' in df.columns:
            model_year_counts = df.groupby(['model', 'released_year']).size()
            validation.update({
                "unique_model_year_combinations": len(model_year_counts),
                "model_year_combinations_with_min_images": {
                    "1+": (model_year_counts >= 1).sum(),
                    "5+": (model_year_counts >= 5).sum(),
                    "10+": (model_year_counts >= 10).sum(),
                    "50+": (model_year_counts >= 50).sum()
                },
                "viewpoint_distribution": df['viewpoint'].value_counts().to_dict()
            })

    return validation