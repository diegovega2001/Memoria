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
from torch.utils.data import Dataset, BatchSampler, DataLoader

from src.defaults import (DEFAULT_VIEWS, DEFAULT_SEED, DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS, DEFAULT_P, DEFAULT_K, DEFAULT_MODEL_TYPE,
                      DEFAULT_DESCRIPTION_INCLUDE, DEFAULT_VERBOSE, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS)
from src.config.TransformConfig import create_standard_transform


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

MODEL_TYPES = {'vision', 'textual', 'both'}
DESCRIPTION_OPTIONS = {'', 'released_year', 'type', 'all'}
UNKNOWN_VALUES = {'unknown', 'Unknown', '', None}

# Configuración fija para muestreo adaptativo
ADAPTIVE_RATIOS = {
    'abundant': {'train': 0.8, 'val': 0.1, 'test': 0.1},
    'few_shot': {'train': 0.7, 'val': 0.15, 'test': 0.15},
    'single_shot': {'train': 0.0, 'val': 0.0, 'test': 1.0}  
}

# Mensajes de error
ERROR_INVALID_MODEL_TYPE = "model_type debe ser uno de: {}"
ERROR_INVALID_DESCRIPTION = "description_include debe ser uno de: {}"


class CarDatasetError(Exception):
    """Excepción personalizada para errores del dataset de vehículos."""
    pass


class CarDataset(Dataset):
    """
    Dataset personalizado para clasificación de vehículos multi-vista con estrategia adaptativa.

    Esta clase maneja datasets de vehículos con múltiples puntos de vista,
    proporcionando funcionalidades de división adaptativa según distribución long-tail,
    augmentación y generación de descripciones textuales para modelos multimodales.

    Attributes:
        df: DataFrame con los datos del dataset.
        views: Lista de vistas/viewpoints a incluir.
        num_views: Número de vistas configuradas.
        min_images_for_abundant_class: Umbral para clasificar clases abundantes.
        seed: Semilla para reproducibilidad.
        transform: Transformaciones a aplicar a las imágenes.
        augment: Si aplicar augmentación de datos.
        model_type: Tipo de modelo ('vision', 'textual', 'both').
        description_include: Información adicional para descripciones.
        model_year_combinations: Lista de tuplas (modelo, año) válidas.
        abundant_models: Modelos con muchas imágenes.
        few_shot_models: Modelos con pocas imágenes.
        single_shot_models: Modelos con muy pocas imágenes.
        label_encoder: Encoder para las etiquetas de modelo-año.
        train_samples: Muestras de entrenamiento.
        val_samples: Muestras de validación.
        test_samples: Muestras de prueba.
        current_split: Split actual ('train', 'val', 'test').
    """

    def __init__(
        self,
        df: pd.DataFrame,
        views: List[str] = DEFAULT_VIEWS,
        min_images_for_abundant_class: int = DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS,
        seed: int = DEFAULT_SEED,
        transform: Optional[Any] = None,
        model_type: str = DEFAULT_MODEL_TYPE,
        description_include: str = DEFAULT_DESCRIPTION_INCLUDE,
        verbose: bool = DEFAULT_VERBOSE
    ) -> None:
        """
        Inicializa el dataset de vehículos con estrategia adaptativa.

        Args:
            df: DataFrame con columnas requeridas: 'model', 'released_year', 'viewpoint', 'image_path'.
            views: Lista de viewpoints a incluir. Si None, usa DEFAULT_VIEWS.
            min_images_for_abundant_class: Umbral para clasificar clases como abundantes.
            seed: Semilla para reproducibilidad aleatoria.
            transform: Transformaciones de torchvision para las imágenes.
            model_type: Tipo de salida ('vision', 'textual', 'both').
            description_include: Información adicional en descripciones.
            verbose: Si mostrar logs detallados durante la inicialización.

        Raises:
            CarDatasetError: Si hay errores de configuración o datos.
            ValueError: Si los parámetros son inválidos.
        """
        # Validación de parámetros
        self._validate_parameters(
            df, views, min_images_for_abundant_class, model_type, description_include
        )

        # Configuración básica
        self.df = df.copy()
        self.views = views if views is not None else DEFAULT_VIEWS.copy()
        self.num_views = len(self.views)
        self.min_images_for_abundant_class = min_images_for_abundant_class
        self.seed = seed
        self.transform = transform 
        self.model_type = model_type
        self.description_include = description_include
        self.verbose = verbose
        self.current_split = 'train'  # Por defecto

        # Configurar random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.verbose:
            logging.info(f"Inicializando CarDataset con {len(self.df)} registros")
            logging.info(f"Vistas configuradas: {self.views}")

        # Inicialización de componentes con estrategia adaptativa
        self._initialize_model_year_combinations()
        self.num_models = len(self.model_year_combinations)
        self.label_encoder = self._initialize_label_encoder()
        self.df = self._filter_dataframe()

        # Creación de splits adaptativos
        self._create_adaptive_data_splits()

        if self.verbose:
            logging.info(f"Dataset inicializado: {self.num_models} combinaciones modelo-año válidas")
            logging.info(f"  - Abundantes: {len(self.abundant_models)}")  
            logging.info(f"  - Few-shot: {len(self.few_shot_models)}")
            logging.info(f"  - Single-shot: {len(self.single_shot_models)}")
            logging.info(f"Samples - Train: {len(self.train_samples)}, Val: {len(self.val_samples)}, Test: {len(self.test_samples)}")

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        views: Optional[List[str]],
        min_images_for_abundant_class: int,
        model_type: str,
        description_include: str
    ) -> None:
        """Valida los parámetros de entrada."""
        # Validar DataFrame
        required_columns = {'model', 'released_year', 'viewpoint', 'image_path'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise CarDatasetError(f"Columnas faltantes en DataFrame: {missing}")

        if df.empty:
            raise CarDatasetError("El DataFrame no puede estar vacío")

        # Validar parámetros numéricos
        if min_images_for_abundant_class <= 0:
            raise ValueError("min_images_for_abundant_class debe ser mayor que 0")

        # Validar opciones categóricas
        if model_type not in MODEL_TYPES:
            raise ValueError(ERROR_INVALID_MODEL_TYPE.format(MODEL_TYPES))

        if description_include not in DESCRIPTION_OPTIONS:
            raise ValueError(ERROR_INVALID_DESCRIPTION.format(DESCRIPTION_OPTIONS))

        # Validar views
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

    def _initialize_model_year_combinations(self) -> None:
        """Identifica combinaciones modelo-año válidas con estrategia adaptativa."""
        if self.verbose:
            logging.info("Identificando combinaciones modelo-año válidas...")

        # Contar imágenes por tupla modelo, año y vista
        counts = self.df.groupby(["model", "released_year", "viewpoint"]).size().unstack(fill_value=0)
        available_views = [v for v in self.views if v in counts.columns]
        
        if not available_views:
            raise CarDatasetError(f"Ninguna de las vistas especificadas existe en el dataset: {self.views}")
        
        # Total de imágenes por combinación modelo-año
        total_counts = counts[available_views].sum(axis=1)
        
        # Clasificar por abundancia
        abundant_candidates = total_counts[total_counts >= self.min_images_for_abundant_class].index.tolist()
        few_shot_candidates = total_counts[(total_counts >= 2) & (total_counts < self.min_images_for_abundant_class)].index.tolist()
        single_shot_candidates = total_counts[total_counts == 1].index.tolist()
        
        if self.verbose:
            logging.info(f"Clasificación inicial:")
            logging.info(f"  - Candidatos abundantes (>={self.min_images_for_abundant_class}): {len(abundant_candidates)}")
            logging.info(f"  - Candidatos few-shot (2-{self.min_images_for_abundant_class-1}): {len(few_shot_candidates)}")
            logging.info(f"  - Candidatos single-shot (1): {len(single_shot_candidates)}")
        
        # Validar que tengan imágenes en las vistas requeridas
        self.abundant_models = []
        self.few_shot_models = []
        self.single_shot_models = []
        
        for model_year_tuple in abundant_candidates:
            if (counts.loc[model_year_tuple][self.views] >= 1).all():
                self.abundant_models.append(model_year_tuple)
        
        for model_year_tuple in few_shot_candidates:
            if (counts.loc[model_year_tuple][self.views] >= 1).all():
                self.few_shot_models.append(model_year_tuple)
        
        for model_year_tuple in single_shot_candidates:
            if (counts.loc[model_year_tuple][self.views] >= 1).all():
                self.single_shot_models.append(model_year_tuple)
        
        if self.verbose:
            logging.info(f"Después del filtrado por vistas:")
            logging.info(f"  - Abundantes finales: {len(self.abundant_models)}")
            logging.info(f"  - Few-shot finales: {len(self.few_shot_models)}")
            logging.info(f"  - Single-shot finales: {len(self.single_shot_models)}")
        
        self.model_year_combinations = self.abundant_models + self.few_shot_models + self.single_shot_models
        
        if not self.model_year_combinations:
            raise CarDatasetError(
                "No se encontraron combinaciones modelo-año válidas con las configuraciones especificadas"
            )

    def _initialize_label_encoder(self) -> LabelEncoder:
        """Inicializa el encoder de etiquetas para las combinaciones modelo-año válidas."""
        label_encoder = LabelEncoder()
        model_year_strings = [f"{model}_{year}" for model, year in self.model_year_combinations]
        label_encoder.fit(model_year_strings)
        if self.verbose:
            logging.info(f"LabelEncoder inicializado con {len(self.model_year_combinations)} clases")
        return label_encoder

    def _filter_dataframe(self) -> pd.DataFrame:
        """Filtra el DataFrame para incluir solo combinaciones modelo-año y vistas válidas."""
        initial_size = len(self.df)
        
        # Crear filtro para combinaciones válidas
        valid_combinations = set(self.model_year_combinations)
        df_with_combinations = self.df.copy()
        df_with_combinations['model_year_tuple'] = list(zip(df_with_combinations['model'], df_with_combinations['released_year']))
        
        filtered_df = df_with_combinations[
            (df_with_combinations['model_year_tuple'].isin(valid_combinations)) &
            (df_with_combinations['viewpoint'].isin(self.views))
        ].drop('model_year_tuple', axis=1)

        final_size = len(filtered_df)
        if self.verbose:
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
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Obtener información de las imágenes
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
        model = rows[0]['model'].strip()
        year = rows[0]['released_year']
        viewpoints = [row['viewpoint'] for row in rows]

        # Construir descripción base
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
        if self.description_include in ['type', 'all']:
            vehicle_type = row.get('type')
            if pd.notna(vehicle_type) and vehicle_type not in UNKNOWN_VALUES:
                desc += f", type {vehicle_type}"

        return desc

    def _create_adaptive_data_splits(self) -> None:
        """Crea las divisiones train/validation/test del dataset con estrategia adaptativa."""
        if self.verbose:
            logging.info("Creando divisiones adaptativas del dataset...")

        train_samples, val_samples, test_samples = [], [], []

        # Procesar cada tipo de modelo con su estrategia específica
        for model_year_tuple in self.abundant_models:
            t, v, te = self._split_abundant_model(model_year_tuple)
            train_samples.extend(t)
            val_samples.extend(v) 
            test_samples.extend(te)
            
        for model_year_tuple in self.few_shot_models:
            t, v, te = self._split_few_shot_model(model_year_tuple)
            train_samples.extend(t)
            val_samples.extend(v)
            test_samples.extend(te)
            
        for model_year_tuple in self.single_shot_models:
            te = self._split_single_shot_model(model_year_tuple)
            test_samples.extend(te)

        # Asignar samples
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

    def _split_abundant_model(self, model_year_tuple: Tuple[str, Any]) -> Tuple[List, List, List]:
        """Split 80/10/10 para modelos abundantes."""
        model, year = model_year_tuple
        model_data = self.df[
            (self.df['model'] == model) & 
            (self.df['released_year'] == year) &
            (self.df['viewpoint'].isin(self.views))
        ]
        
        # Agrupar por vista y obtener paths
        grouped = model_data.groupby('viewpoint')
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
        if min_images == 0:
            return [], [], []
            
        # Aplicar ratios 80/10/10
        ratios = ADAPTIVE_RATIOS['abundant']
        n_train = max(1, int(min_images * ratios['train']))
        n_val = max(1, int(min_images * ratios['val']))
        n_test = min_images - n_train - n_val
        n_test = max(1, n_test)
        
        return self._create_samples_for_splits(view_images, model_year_tuple, n_train, n_val, n_test, min_images)

    def _split_few_shot_model(self, model_year_tuple: Tuple[str, Any]) -> Tuple[List, List, List]:
        """Split 70/15/15 para modelos few-shot."""
        model, year = model_year_tuple
        model_data = self.df[
            (self.df['model'] == model) & 
            (self.df['released_year'] == year) &
            (self.df['viewpoint'].isin(self.views))
        ]
        
        grouped = model_data.groupby('viewpoint')
        view_images = {}
        for view in self.views:
            if view in grouped.groups:
                paths = list(grouped.get_group(view)['image_path'])
                random.shuffle(paths)
                view_images[view] = paths
            else:
                view_images[view] = []
                
        min_images = min(len(view_images[view]) for view in self.views if len(view_images[view]) > 0)
        if min_images == 0:
            return [], [], []
            
        # Aplicar ratios 70/15/15
        ratios = ADAPTIVE_RATIOS['few_shot']
        if min_images >= 3:
            n_train = max(1, int(min_images * ratios['train']))
            n_val = max(1, int(min_images * ratios['val']))
            n_test = min_images - n_train - n_val
            n_test = max(1, n_test)
        else:
            # Para 2 imágenes: 1 train, 0 val, 1 test
            n_train = 1
            n_val = 0
            n_test = 1
            
        return self._create_samples_for_splits(view_images, model_year_tuple, n_train, n_val, n_test, min_images)

    def _split_single_shot_model(self, model_year_tuple: Tuple[str, Any]) -> List:
        """Solo test para modelos single-shot."""
        model, year = model_year_tuple
        model_data = self.df[
            (self.df['model'] == model) & 
            (self.df['released_year'] == year) &
            (self.df['viewpoint'].isin(self.views))
        ]
        
        grouped = model_data.groupby('viewpoint')
        view_images = []
        for view in self.views:
            if view in grouped.groups:
                paths = list(grouped.get_group(view)['image_path'])
                view_images.append(paths)
            else:
                view_images.append([])
                
        # Crear sample con las imágenes disponibles
        test_samples = []
        max_samples = max(len(paths) for paths in view_images)
        for i in range(max_samples):
            image_pair = [view_images[j][i] for j in range(len(self.views)) if i < len(view_images[j])]
            if image_pair:  # Al menos una imagen disponible
                if self.model_type in ['textual', 'both']:
                    text_desc = self._create_text_descriptor(image_pair)
                    test_samples.append((model_year_tuple, image_pair, text_desc))
                else:
                    test_samples.append((model_year_tuple, image_pair))
                
        return test_samples

    def _create_samples_for_splits(
        self, 
        view_images: Dict[str, List[str]], 
        model_year_tuple: Tuple[str, Any],
        n_train: int, 
        n_val: int,
        n_test,
        min_images: int
    ) -> Tuple[List, List, List]:
        """Crea muestras para train/val/test."""
        train_samples = []
        val_samples = []
        test_samples = []
        
        for i in range(min_images):
            image_pair = [view_images[view][i] for view in self.views if i < len(view_images[view])]
            if len(image_pair) == len(self.views):  # Solo si hay imagen para cada vista
                if self.model_type in ['textual', 'both']:
                    text_desc = self._create_text_descriptor(image_pair)
                    sample = (model_year_tuple, image_pair, text_desc)
                else:
                    sample = (model_year_tuple, image_pair)
                
                if i < n_train:
                    train_samples.append(sample)
                elif i < n_train + n_val and n_val > 0:
                    val_samples.append(sample)
                else:
                    test_samples.append(sample)
                    
        return train_samples, val_samples, test_samples

    def set_split(self, split: str) -> None:
        """
        Cambia el split actual del dataset.

        Args:
            split: 'train', 'val' o 'test'.

        Raises:
            ValueError: Si el split no es válido.
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError("split debe ser 'train', 'val' o 'test'")
        
        self.current_split = split
        if self.verbose:
            logging.info(f"Dataset configurado para split: {split}")

    def get_current_samples(self) -> List:
        """Obtiene las muestras del split actual."""
        if self.current_split == 'train':
            return self.train_samples
        elif self.current_split == 'val':
            return self.val_samples
        elif self.current_split == 'test':
            return self.test_samples
        else:
            raise ValueError(f"Split no válido: {self.current_split}")

    def __len__(self) -> int:
        """Retorna el número de muestras del split actual."""
        return len(self.get_current_samples())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Obtiene una muestra del dataset según el split actual.

        Args:
            idx: Índice de la muestra.

        Returns:
            Diccionario con imágenes, etiquetas y descripción textual (si aplica).

        Raises:
            FileNotFoundError: Si alguna imagen no se puede cargar.
            Exception: Para otros errores de procesamiento.
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
                for img, bbox in images:
                    if hasattr(self.transform, 'use_bbox') and self.transform.use_bbox and bbox is not None:
                        processed_img = self.transform(img, bbox=bbox)
                    else:
                        processed_img = self.transform(img)
                    processed_images.append(processed_img)
                images = processed_images
            else:
                # Si no hay transformaciones, extraer solo las imágenes
                images = [img for img, _ in images]

            # Crear etiqueta
            model_year_string = f"{model_year_tuple[0]}_{model_year_tuple[1]}"
            label = torch.tensor(
                self.label_encoder.transform([model_year_string])[0], 
                dtype=torch.long
            )

            # Preparar salida
            # Convertir lista de imágenes a tensor
            if len(images) == 1:
                image_tensor = images[0]  # Tensor único
            else:
                image_tensor = torch.stack(images)  # Stack múltiples tensores

            output = {
                "images": image_tensor,
                "labels": label,
                "model_name": model_year_string
            }

            if text_desc is not None:
                output["text_description"] = text_desc

            return output

        except Exception as e:
            logging.error(f"Error cargando muestra {idx} del split {self.current_split}: {e}")
            raise

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del dataset."""
        # Calcular estadísticas por split
        train_stats = self._calculate_split_stats(self.train_samples, "train")
        val_stats = self._calculate_split_stats(self.val_samples, "validation")
        test_stats = self._calculate_split_stats(self.test_samples, "test")

        return {
            "overview": {
                "num_model_year_combinations": self.num_models,
                "num_views": self.num_views,
                "views": self.views,
                "min_images_for_abundant_class": self.min_images_for_abundant_class,
                "model_type": self.model_type,
                "adaptive_strategy": {
                    "abundant": len(self.abundant_models),
                    "few_shot": len(self.few_shot_models),
                    "single_shot": len(self.single_shot_models)
                }
            },
            "splits": {
                "train": train_stats,
                "validation": val_stats,
                "test": test_stats
            },
            "sample_combinations": self.model_year_combinations[:10],  # Primeros 10
            "total_combinations": len(self.model_year_combinations)
        }

    def _calculate_split_stats(self, samples: List, split_name: str) -> Dict:
        """Calcula estadísticas para una división del dataset."""
        if len(samples) == 0:
            return {"total_samples": 0}

        samples_per_combination = []
        for model_year_tuple in self.model_year_combinations:
            count = len([s for s in samples if s[0] == model_year_tuple])
            samples_per_combination.append(count)

        samples_array = np.array(samples_per_combination)
        
        return {
            "total_samples": len(samples),
            "samples_per_combination_mean": float(samples_array.mean()),
            "samples_per_combination_std": float(samples_array.std()),
            "samples_per_combination_min": int(samples_array.min()),
            "samples_per_combination_max": int(samples_array.max())
        }

    def __str__(self) -> str:
        """Representación string detallada del dataset."""
        lines = ["=== Car Dataset Overview (Adaptive Strategy) ==="]
        lines.append(f"Views: {self.views}")
        lines.append(f"Number of model-year combinations: {self.num_models}")
        lines.append(f"Min images for abundant class: {self.min_images_for_abundant_class}")
        lines.append(f"Model type: {self.model_type}")
        lines.append(f"Description includes: {self.description_include or 'basic info only'}")
        lines.append(f"Current split: {self.current_split}")
        lines.append("")
        lines.append("Distribution strategy:")
        lines.append(f"  - Abundant models (≥{self.min_images_for_abundant_class} imgs): {len(self.abundant_models)}")
        lines.append(f"  - Few-shot models (2-{self.min_images_for_abundant_class-1} imgs): {len(self.few_shot_models)}")
        lines.append(f"  - Single-shot models (1 img): {len(self.single_shot_models)}")
        lines.append("")
        
        # Estadísticas de divisiones
        splits_data = [
            ("Train", self.train_samples),
            ("Validation", self.val_samples), 
            ("Test", self.test_samples)
        ]
        
        for split_name, samples in splits_data:
            if len(samples) > 0:
                stats = self._calculate_split_stats(samples, split_name.lower())
                lines.append(f"{split_name} split:")
                lines.append(f"  Total samples: {stats['total_samples']}")
                lines.append(f"  Samples per combination - Mean: {stats['samples_per_combination_mean']:.1f}, "
                           f"Std: {stats['samples_per_combination_std']:.1f}")
            else:
                lines.append(f"{split_name} split: 0 samples")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representación concisa para debugging."""
        return (f"CarDataset(combinations={self.num_models}, "
                f"views={len(self.views)}, strategy=adaptive, split={self.current_split})")


class IdentitySampler(BatchSampler):
    """
    BatchSampler P×K para contrastive learning.
    
    Este sampler crea batches con P clases y K muestras por clase,
    útil para entrenamiento con pérdidas contrastivas donde necesitas
    múltiples ejemplos de la misma clase en cada batch.
    
    Args:
        samples: Lista de muestras del dataset de entrenamiento.
        P: Número de clases por batch.
        K: Número de muestras por clase en cada batch.
        seed: Semilla para reproducibilidad.
    """
    
    def __init__(
        self, 
        samples: List[Tuple], 
        P: int = DEFAULT_P, 
        K: int = DEFAULT_K, 
        seed: int = DEFAULT_SEED
    ):
        self.samples = samples
        self.P = P
        self.K = K
        self.seed = seed
        
        # Agrupar índices por clase (modelo_año_tuple)
        self.class_to_indices = {}
        for idx, sample in enumerate(samples):
            model_year_tuple = sample[0]  # (model, year) tuple
            if model_year_tuple not in self.class_to_indices:
                self.class_to_indices[model_year_tuple] = []
            self.class_to_indices[model_year_tuple].append(idx)
            
        self.classes = list(self.class_to_indices.keys())
        self.batch_size = self.P * self.K
        
        # Filtrar clases que tienen al menos K muestras
        valid_classes = [
            cls for cls in self.classes 
            if len(self.class_to_indices[cls]) >= self.K
        ]
        
        if len(valid_classes) < self.P:
            logging.warning(f"Solo {len(valid_classes)} clases tienen ≥{self.K} muestras. "
                          f"Se necesitan al menos {self.P} para el sampler P×K.")
            # Usar todas las clases disponibles si no hay suficientes
            self.valid_classes = self.classes
            self.P = min(self.P, len(self.classes))
        else:
            self.valid_classes = valid_classes
        
        # Calcular número de batches posibles
        min_samples_per_class = min(
            len(self.class_to_indices[cls]) for cls in self.valid_classes
        ) if self.valid_classes else 0
        
        self.num_batches = max(1, min_samples_per_class // self.K * len(self.valid_classes) // self.P)
        
        logging.info(f"IdentitySampler configurado: P={self.P}, K={self.K}, "
                    f"clases válidas={len(self.valid_classes)}, batches={self.num_batches}")
        
    def __iter__(self):
        random.seed(self.seed)
        
        for batch_num in range(self.num_batches):
            batch_indices = []
            
            # Seleccionar P clases aleatoriamente
            if len(self.valid_classes) >= self.P:
                selected_classes = random.sample(self.valid_classes, self.P)
            else:
                # Si no hay suficientes clases válidas, usar las que hay
                selected_classes = random.choices(self.valid_classes, k=self.P)
            
            for cls in selected_classes:
                class_indices = self.class_to_indices[cls].copy()
                random.shuffle(class_indices)
                
                # Tomar K muestras de esta clase
                selected = class_indices[:min(self.K, len(class_indices))]
                batch_indices.extend(selected)
                
                # Si la clase no tiene suficientes muestras, rellenar con repeticiones
                while len(selected) < self.K and class_indices:
                    additional = random.choice(class_indices)
                    selected.append(additional)
                    batch_indices.append(additional)
                    if len(selected) >= self.K:
                        break
            
            # Asegurar tamaño de batch correcto
            while len(batch_indices) < self.batch_size:
                random_class = random.choice(self.valid_classes)
                random_idx = random.choice(self.class_to_indices[random_class])
                batch_indices.append(random_idx)
                
            yield batch_indices[:self.batch_size]
            
    def __len__(self):
        return self.num_batches


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
    **dataset_kwargs
) -> Dict[str, Any]:
    """
    Crea dataset con estrategia adaptativa y DataLoaders listos para usar.
    
    Args:
        df: DataFrame con datos del dataset.
        views: Lista de vistas a incluir.
        min_images_for_abundant_class: Umbral para clases abundantes.
        P: Número de clases por batch para contrastive learning (solo train).
        K: Número de muestras por clase por batch (solo train).
        train_transform: Transformaciones para train (CON augmentación recomendado).
        val_transform: Transformaciones para val/test (SIN augmentación).
                      Si es None, usa train_transform sin augmentación.
        batch_size: Tamaño de batch para val y test.
        num_workers: Número de workers para DataLoaders.
        seed: Semilla para reproducibilidad.
        **dataset_kwargs: Argumentos adicionales para CarDataset.
    
    Returns:
        Diccionario con 'dataset', 'train_loader', 'val_loader', 'test_loader', 'train_sampler'.
        
    Example:
        >>> from src.config.TransformConfig import create_training_transform
        >>> transform = create_training_transform(size=(224, 224), use_bbox=True)
        >>> dataset_dict = create_car_dataset(df=datafram, views=['front', 'rear'], train_transform=train_transform, val_transform=val_transform)
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
    
    # Train loader con IdentitySampler P×K
    train_sampler = IdentitySampler(base_dataset.train_samples, P=P, K=K, seed=seed)
    
    # Crear datasets para cada split con transforms específicos
    train_dataset = copy.deepcopy(base_dataset)
    train_dataset.transform = create_standard_transform(augment=True)  # Transform CON augmentación
    train_dataset.verbose = False
    train_dataset.set_split('train')
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Val loader
    val_dataset = copy.deepcopy(base_dataset)
    val_dataset.transform = create_standard_transform(augment=False)  # Transform SIN augmentación
    val_dataset.verbose = False
    val_dataset.set_split('val')
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader
    test_dataset = copy.deepcopy(base_dataset)
    test_dataset.transform = create_standard_transform(augment=False)  # Transform SIN augmentación
    test_dataset.set_split('test')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logging.info(f"DataLoaders creados:")
    logging.info(f"  - Train: {len(train_loader)} batches de tamaño {P}×{K}={P*K}")
    logging.info(f"  - Val: {len(val_loader)} batches de tamaño {batch_size}")
    logging.info(f"  - Test: {len(test_loader)} batches de tamaño {batch_size}")
    
    # Log sobre augmentación
    train_has_augment = hasattr(train_transform, 'augment') and train_transform.augment
    val_has_augment = hasattr(val_transform, 'augment') and val_transform.augment
    logging.info(f"  - Train augmentación: {'Habilitada' if train_has_augment else 'Deshabilitada'}")
    logging.info(f"  - Val/Test augmentación: {'Habilitada' if val_has_augment else 'Deshabilitada'}")
    
    return {
        "dataset": base_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_sampler": train_sampler
    }


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