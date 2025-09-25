"""
Módulo de dataset personalizado para clasificación de vehículos multi-vista.

Este módulo proporciona clases para manejar datasets de vehículos con múltiples
vistas, incluyendo funcionalidades de división train/val/test, augmentación
y generación de descripciones textuales.
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constantes del dataset
DEFAULT_VIEWS = ['front', 'rear']
DEFAULT_SEED = 3
MODEL_TYPES = {'vision', 'textual', 'both'}
DESCRIPTION_OPTIONS = {'', 'released_year', 'type', 'all'}
UNKNOWN_VALUES = {'unknown', 'Unknown', '', None}

# Mensajes de error
ERROR_INVALID_MODEL_TYPE = "model_type debe ser uno de: {}"
ERROR_INVALID_DESCRIPTION = "description_include debe ser uno de: {}"
ERROR_INVALID_RATIOS = "La suma de val_ratio y test_ratio no puede ser mayor a 1.0"
ERROR_INSUFFICIENT_IMAGES = "No hay suficientes imágenes para el modelo {}: {}"


class CarDatasetError(Exception):
    """Excepción personalizada para errores del dataset de vehículos."""
    pass


class CarDataset(Dataset):
    """
    Dataset personalizado para clasificación de vehículos multi-vista.

    Esta clase maneja datasets de vehículos con múltiples puntos de vista,
    proporcionando funcionalidades de división automática, augmentación
    y generación de descripciones textuales para modelos multimodales.

    Attributes:
        df: DataFrame con los datos del dataset.
        views: Lista de vistas/viewpoints a incluir.
        num_views: Número de vistas configuradas.
        train_images: Número de imágenes de entrenamiento por modelo por vista.
        val_ratio: Proporción de imágenes para validación.
        test_ratio: Proporción de imágenes para prueba.
        seed: Semilla para reproducibilidad.
        transform: Transformaciones a aplicar a las imágenes.
        augment: Si aplicar augmentación de datos.
        model_type: Tipo de modelo ('vision', 'textual', 'both').
        description_include: Información adicional para descripciones.
        models: Lista de modelos válidos en el dataset.
        label_encoder: Encoder para las etiquetas de modelos.
        train: Dataset de entrenamiento.
        val: Dataset de validación.
        test: Dataset de prueba.

    Example:
        >>> df = pd.read_csv('cars_dataset.csv')
        >>> dataset = CarDataset(
        ...     df=df,
        ...     views=['front', 'rear'],
        ...     train_images=50,
        ...     val_ratio=0.2,
        ...     test_ratio=0.2
        ... )
        >>> train_loader = DataLoader(dataset.train, batch_size=32)

    Raises:
        CarDatasetError: Para errores específicos del dataset.
        ValueError: Para parámetros inválidos.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        views: List[str] = None,
        train_images: int = 50,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = DEFAULT_SEED,
        transform: Optional[Any] = None,
        augment: bool = False,
        model_type: str = 'vision',
        description_include: str = ''
    ) -> None:
        """
        Inicializa el dataset de vehículos.

        Args:
            df: DataFrame con columnas requeridas: 'model', 'viewpoint', 'image_path'.
            views: Lista de viewpoints a incluir. Si None, usa DEFAULT_VIEWS.
            train_images: Número mínimo de imágenes de entrenamiento por modelo por vista.
            val_ratio: Proporción de datos para validación (0.0-1.0).
            test_ratio: Proporción de datos para prueba (0.0-1.0).
            seed: Semilla para reproducibilidad aleatoria.
            transform: Transformaciones de torchvision para las imágenes.
            augment: Si aplicar augmentación horizontal flip aleatoria.
            model_type: Tipo de salida ('vision', 'textual', 'both').
            description_include: Información adicional en descripciones.

        Raises:
            CarDatasetError: Si hay errores de configuración o datos.
            ValueError: Si los parámetros son inválidos.
        """
        # Validación de parámetros
        self._validate_parameters(
            df, views, train_images, val_ratio, test_ratio, model_type, description_include
        )

        # Configuración básica
        self.df = df.copy()
        self.views = views if views is not None else DEFAULT_VIEWS.copy()
        self.num_views = len(self.views)
        self.train_images = train_images
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.transform = transform
        self.augment = augment
        self.model_type = model_type
        self.description_include = description_include

        # Configurar random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        logging.info(f"Inicializando CarDataset con {len(self.df)} registros")
        logging.info(f"Vistas configuradas: {self.views}")

        # Inicialización de componentes
        self.models = self._initialize_valid_models()
        self.num_models = len(self.models)
        self.label_encoder = self._initialize_label_encoder()
        self.df = self._filter_dataframe()

        # Creación de splits
        self.train, self.val, self.test = self._create_data_splits()

        logging.info(f"Dataset inicializado: {self.num_models} modelos válidos")

    def _validate_parameters(
        self,
        df: pd.DataFrame,
        views: Optional[List[str]],
        train_images: int,
        val_ratio: float,
        test_ratio: float,
        model_type: str,
        description_include: str
    ) -> None:
        """Valida los parámetros de entrada."""
        # Validar DataFrame
        required_columns = {'model', 'viewpoint', 'image_path'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise CarDatasetError(f"Columnas faltantes en DataFrame: {missing}")

        if df.empty:
            raise CarDatasetError("El DataFrame no puede estar vacío")

        # Validar parámetros numéricos
        if train_images <= 0:
            raise ValueError("train_images debe ser mayor que 0")

        if not (0.0 <= val_ratio <= 1.0):
            raise ValueError("val_ratio debe estar entre 0.0 y 1.0")

        if not (0.0 <= test_ratio <= 1.0):
            raise ValueError("test_ratio debe estar entre 0.0 y 1.0")

        if val_ratio + test_ratio > 1.0:
            raise ValueError(ERROR_INVALID_RATIOS)

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

    def _initialize_valid_models(self) -> List[str]:
        """
        Identifica modelos que tienen suficientes imágenes en todas las vistas.

        Returns:
            Lista de modelos válidos.

        Raises:
            CarDatasetError: Si no hay modelos válidos.
        """
        logging.info("Identificando modelos válidos...")

        # Contar imágenes por modelo y vista
        counts = (
            self.df.groupby(["model", "viewpoint"])
            .size()
            .unstack(fill_value=0)
        )

        # Filtrar modelos con suficientes imágenes en todas las vistas
        valid_models = counts[
            (counts[self.views] >= self.train_images).all(axis=1)
        ].index.tolist()

        if not valid_models:
            raise CarDatasetError(
                f"No se encontraron modelos con al menos {self.train_images} "
                f"imágenes en todas las vistas: {self.views}"
            )

        # Log de estadísticas
        total_models = len(counts.index)
        logging.info(f"Modelos válidos: {len(valid_models)}/{total_models}")

        # Log de modelos filtrados con razones
        filtered_models = set(counts.index) - set(valid_models)
        if filtered_models:
            logging.warning(f"Modelos filtrados por insuficientes imágenes: {len(filtered_models)}")
            for model in list(filtered_models)[:5]:  # Log primeros 5
                model_counts = counts.loc[model][self.views].to_dict()
                logging.debug(f"  {model}: {model_counts}")

        return valid_models

    def _initialize_label_encoder(self) -> LabelEncoder:
        """Inicializa el encoder de etiquetas para los modelos válidos."""
        label_encoder = LabelEncoder()
        label_encoder.fit(self.models)
        logging.info(f"LabelEncoder inicializado con {len(self.models)} clases")
        return label_encoder

    def _filter_dataframe(self) -> pd.DataFrame:
        """
        Filtra el DataFrame para incluir solo modelos y vistas válidos.

        Returns:
            DataFrame filtrado.
        """
        initial_size = len(self.df)
        
        filtered_df = self.df[
            (self.df["model"].isin(self.models)) &
            (self.df["viewpoint"].isin(self.views))
        ].copy()

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
        viewpoints = [row['viewpoint'] for row in rows]

        # Construir descripción base
        if len(viewpoints) == 1:
            desc = f"The {viewpoints[0]} view image of a {make} {model} vehicle"
        else:
            viewpoint_text = " and ".join(viewpoints)
            desc = f"The {viewpoint_text} view images of a {make} {model} vehicle"

        # Agregar información adicional según configuración
        desc = self._add_additional_info(desc, rows[0])
        
        return desc + "."

    def _add_additional_info(self, desc: str, row: pd.Series) -> str:
        """Agrega información adicional a la descripción."""
        if self.description_include in ['released_year', 'all']:
            released_year = row.get('released_year')
            if pd.notna(released_year) and str(released_year) not in UNKNOWN_VALUES:
                desc += f", year {released_year}"

        if self.description_include in ['type', 'all']:
            vehicle_type = row.get('type')
            if pd.notna(vehicle_type) and vehicle_type not in UNKNOWN_VALUES:
                desc += f", type {vehicle_type}"

        return desc

    def _create_data_splits(self) -> Tuple[SplitDataset, SplitDataset, SplitDataset]:
        """
        Crea las divisiones train/validation/test del dataset.

        Returns:
            Tupla con (train_dataset, val_dataset, test_dataset).

        Raises:
            CarDatasetError: Si hay errores creando las divisiones.
        """
        logging.info("Creando divisiones del dataset...")

        train_samples, val_samples, test_samples = [], [], []

        for model in self.models:
            try:
                # Obtener imágenes del modelo para cada vista
                model_data = self.df[
                    (self.df['model'] == model) & 
                    (self.df['viewpoint'].isin(self.views))
                ]
                
                grouped = model_data.groupby('viewpoint')
                view_images = {}
                
                for view in self.views:
                    if view in grouped.groups:
                        view_images[view] = list(grouped.get_group(view)['image_path'])
                    else:
                        view_images[view] = []

                # Verificar que hay suficientes imágenes
                min_images = min(len(view_images[view]) for view in self.views)
                if min_images < self.train_images:
                    raise CarDatasetError(
                        ERROR_INSUFFICIENT_IMAGES.format(model, min_images)
                    )

                # Balancear y mezclar imágenes por vista
                random.seed(self.seed)
                for view in self.views:
                    random.shuffle(view_images[view])
                    view_images[view] = view_images[view][:min_images]

                # Crear divisiones
                model_train, model_val, model_test = self._split_model_data(
                    view_images, model
                )
                
                train_samples.extend(model_train)
                val_samples.extend(model_val)
                test_samples.extend(model_test)

            except Exception as e:
                raise CarDatasetError(f"Error procesando modelo {model}: {e}") from e

        # Crear datasets
        train_dataset = SplitDataset(
            train_samples, self.label_encoder, self.df, self.transform, self.augment
        )
        val_dataset = SplitDataset(
            val_samples, self.label_encoder, self.df, self.transform, augment=False
        )
        test_dataset = SplitDataset(
            test_samples, self.label_encoder, self.df, self.transform, augment=False
        )

        # Log de estadísticas
        logging.info(f"Divisiones creadas - Train: {len(train_dataset)}, "
                    f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def _split_model_data(
        self, 
        view_images: Dict[str, List[str]], 
        model: str
    ) -> Tuple[List, List, List]:
        """Divide los datos de un modelo específico en train/val/test."""
        # Separar imágenes de entrenamiento
        train_paths = [view_images[view][:self.train_images] for view in self.views]
        remaining_paths = [view_images[view][self.train_images:] for view in self.views]

        # Calcular divisiones de validación y prueba
        val_paths, test_paths = [], []
        
        if self.val_ratio + self.test_ratio > 0:
            n_remaining = len(remaining_paths[0])
            n_val = int(n_remaining * self.val_ratio) if self.val_ratio > 0 else 0
            
            val_paths = [paths[:n_val] for paths in remaining_paths]
            test_paths = [paths[n_val:] for paths in remaining_paths]
        else:
            val_paths = [[] for _ in self.views]
            test_paths = remaining_paths

        # Crear muestras con descripciones si es necesario
        train_samples = self._create_samples(model, train_paths)
        val_samples = self._create_samples(model, val_paths)
        test_samples = self._create_samples(model, test_paths)

        return train_samples, val_samples, test_samples

    def _create_samples(self, model: str, view_paths: List[List[str]]) -> List:
        """Crea muestras para un conjunto de rutas."""
        samples = []
        max_samples = max((len(paths) for paths in view_paths), default=0)
        
        for i in range(max_samples):
            # Crear par de imágenes para todas las vistas
            image_pair = [view_paths[j][i] for j in range(len(self.views)) if i < len(view_paths[j])]
            
            if len(image_pair) == len(self.views):  # Solo si hay imagen para cada vista
                if self.model_type in ['textual', 'both']:
                    text_desc = self._create_text_descriptor(image_pair)
                    samples.append((model, image_pair, text_desc))
                else:
                    samples.append((model, image_pair))
        
        return samples

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del dataset.

        Returns:
            Diccionario con estadísticas completas.
        """
        if not hasattr(self, 'train'):
            return {"error": "Dataset no inicializado"}

        # Calcular estadísticas por modelo
        train_stats = self._calculate_split_stats(self.train, "train")
        val_stats = self._calculate_split_stats(self.val, "validation")
        test_stats = self._calculate_split_stats(self.test, "test")

        return {
            "overview": {
                "num_models": self.num_models,
                "num_views": self.num_views,
                "views": self.views,
                "train_images_per_model": self.train_images,
                "model_type": self.model_type
            },
            "splits": {
                "train": train_stats,
                "validation": val_stats,
                "test": test_stats
            },
            "models": self.models[:10],  # Primeros 10 modelos
            "total_models": len(self.models)
        }

    def _calculate_split_stats(self, dataset: SplitDataset, split_name: str) -> Dict:
        """Calcula estadísticas para una división del dataset."""
        if len(dataset) == 0:
            return {"total_samples": 0}

        samples_per_model = []
        for model in self.models:
            count = len([s for s in dataset.samples if s[0] == model])
            samples_per_model.append(count)

        samples_array = np.array(samples_per_model)
        
        return {
            "total_samples": len(dataset),
            "samples_per_model_mean": float(samples_array.mean()),
            "samples_per_model_std": float(samples_array.std()),
            "samples_per_model_min": int(samples_array.min()),
            "samples_per_model_max": int(samples_array.max())
        }

    def __str__(self) -> str:
        """Representación string detallada del dataset."""
        if not hasattr(self, 'train'):
            return "CarDataset(no inicializado)"

        lines = ["=== Car Dataset Overview ==="]
        lines.append(f"Views: {self.views}")
        lines.append(f"Number of models: {self.num_models}")
        lines.append(f"Train images per model per view: {self.train_images}")
        lines.append(f"Val ratio: {self.val_ratio}, Test ratio: {self.test_ratio}")
        lines.append(f"Model type: {self.model_type}")
        lines.append(f"Description includes: {self.description_include or 'basic info only'}")
        lines.append("")
        
        # Estadísticas de divisiones
        for split_name, dataset in [("Train", self.train), ("Validation", self.val), ("Test", self.test)]:
            if len(dataset) > 0:
                stats = self._calculate_split_stats(dataset, split_name.lower())
                lines.append(f"{split_name} split:")
                lines.append(f"  Total samples: {stats['total_samples']}")
                lines.append(f"  Samples per model - Mean: {stats['samples_per_model_mean']:.1f}, "
                           f"Std: {stats['samples_per_model_std']:.1f}")
            else:
                lines.append(f"{split_name} split: 0 samples")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Representación concisa para debugging."""
        status = "inicializado" if hasattr(self, 'train') else "no inicializado"
        return (f"CarDataset(models={self.num_models if hasattr(self, 'num_models') else 'N/A'}, "
                f"views={len(self.views)}, status={status})")


class SplitDataset(Dataset):
    """
    Dataset para una división específica (train/val/test).

    Esta clase maneja las muestras de una división específica del dataset,
    aplicando transformaciones y augmentación según corresponda.

    Attributes:
        samples: Lista de muestras (model, paths, [text_desc]).
        label_encoder: Encoder para convertir modelos a etiquetas numéricas.
        transform: Transformaciones a aplicar a las imágenes.
        augment: Si aplicar augmentación de datos.
    """

    def __init__(
        self,
        samples: List[Tuple],
        label_encoder: LabelEncoder,
        df: pd.DataFrame,
        transform: Optional[Any] = None,
        augment: bool = False
    ) -> None:
        """
        Inicializa el dataset de división.

        Args:
            samples: Lista de muestras del dataset.
            label_encoder: Encoder de etiquetas preentrenado.
            df: DataFrame con información completa incluyendo bounding boxes.
            transform: Transformaciones para las imágenes.
            augment: Si aplicar augmentación horizontal flip.
        """
        self.samples = samples
        self.label_encoder = label_encoder
        self.df = df
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        """Retorna el número de muestras."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Obtiene una muestra del dataset.

        Args:
            idx: Índice de la muestra.

        Returns:
            Diccionario con imágenes, etiquetas y descripción textual (si aplica).

        Raises:
            FileNotFoundError: Si alguna imagen no se puede cargar.
            Exception: Para otros errores de procesamiento.
        """
        try:
            sample = self.samples[idx]
            
            # Extraer información de la muestra
            if len(sample) == 3:
                model, paths, text_desc = sample
            else:
                model, paths = sample
                text_desc = None

            # Cargar imágenes con bounding boxes
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
                            bbox = matching_rows.iloc[0]['bbox']
                    
                    images.append((img, bbox))
                except Exception as e:
                    raise FileNotFoundError(f"No se pudo cargar imagen {path}: {e}") from e

            # Aplicar augmentación si está habilitada
            if self.augment:
                images = [
                    (F.hflip(img) if random.random() > 0.5 else img, bbox)
                    for img, bbox in images
                ]

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
            label = torch.tensor(
                self.label_encoder.transform([model])[0], 
                dtype=torch.long
            )

            # Preparar salida
            output = {
                "images": images,
                "labels": label
            }

            if text_desc is not None:
                output["text_description"] = text_desc

            return output

        except Exception as e:
            logging.error(f"Error cargando muestra {idx}: {e}")
            raise

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Obtiene información de una muestra sin cargar las imágenes.

        Args:
            idx: Índice de la muestra.

        Returns:
            Diccionario con información de la muestra.
        """
        if idx >= len(self.samples):
            raise IndexError(f"Índice fuera de rango: {idx}")

        sample = self.samples[idx]
        
        info = {
            "index": idx,
            "model": sample[0],
            "image_paths": sample[1],
            "num_images": len(sample[1])
        }

        if len(sample) == 3:
            info["text_description"] = sample[2]

        return info


# Funciones de conveniencia
def create_car_dataset(
    df: pd.DataFrame,
    views: List[str] = None,
    train_images: int = 5,
    val_ratio: float = 0.5,
    test_ratio: float = 0.5,
    **kwargs
) -> CarDataset:
    """
    Función de conveniencia para crear un dataset de vehículos.

    Args:
        df: DataFrame con los datos.
        views: Vistas a incluir.
        train_images: Imágenes de entrenamiento por modelo.
        split_ratios: Tupla con (val_ratio, test_ratio).
        **kwargs: Argumentos adicionales para CarDataset.

    Returns:
        CarDataset configurado.
    """

    return CarDataset(
        df=df,
        views=views,
        train_images=train_images,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        **kwargs
    )


def validate_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida la estructura de un DataFrame para usar con CarDataset.

    Args:
        df: DataFrame a validar.

    Returns:
        Diccionario con información de validación.
    """
    required_columns = {'model', 'viewpoint', 'image_path'}
    missing_columns = required_columns - set(df.columns)
    
    validation = {
        "valid": len(missing_columns) == 0,
        "missing_columns": list(missing_columns),
        "total_records": len(df),
        "unique_models": df['model'].nunique() if 'model' in df.columns else 0,
        "unique_viewpoints": df['viewpoint'].nunique() if 'viewpoint' in df.columns else 0,
        "available_viewpoints": df['viewpoint'].unique().tolist() if 'viewpoint' in df.columns else []
    }

    if validation["valid"]:
        # Estadísticas adicionales si es válido
        model_counts = df['model'].value_counts()
        validation.update({
            "models_with_min_images": {
                "10+": (model_counts >= 10).sum(),
                "50+": (model_counts >= 50).sum(),
                "100+": (model_counts >= 100).sum()
            },
            "viewpoint_distribution": df['viewpoint'].value_counts().to_dict()
        })

    return validation