"""
Módulo de generación de dataset CSV para CompCars dataset.

Este módulo proporciona una clase para construir y guardar un dataset
a partir de los datos del "The Comprehensive Cars (CompCars) dataset",
procesando imágenes, metadatos y etiquetas para crear un CSV estructurado.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import warnings

import pandas as pd
import scipy.io

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constantes del dataset
DEFAULT_VIEWPOINT_FILTER = {1, 2}  # front y rear
VIEWPOINT_MAPPING = {1: 'front', 2: 'rear', 3: 'side', 4: 'frontside', 5: 'rearside'}
UNKNOWN_TYPE = 'Unknown'
DEFAULT_COLUMNS_TO_KEEP = [
    'image_name', 'image_path', 'released_year', 'viewpoint',
    'bbox', 'make', 'model', 'type'
]

# Nombres de archivos del dataset
ATTRIBUTES_FILE = 'attributes.txt'
MAKE_MODEL_FILE = 'make_model_name.mat'
CAR_TYPE_FILE = 'car_type.mat'


class CompCarsDatasetError(Exception):
    """Excepción personalizada para errores relacionados con el dataset CompCars."""
    pass


class DataFrameMaker:
    """
    Generador de dataset CSV a partir del CompCars dataset.

    Esta clase permite configurar, procesar y generar un archivo CSV que contenga
    los datos relevantes del dataset CompCars, incluyendo metadatos de vehículos,
    rutas de imágenes y información de bounding boxes.

    Attributes:
        base_path: Ruta base donde está almacenado el dataset CompCars.
        images_folder: Carpeta que contiene las imágenes del dataset.
        labels_folder: Carpeta que contiene las etiquetas/anotaciones.
        misc_folder: Carpeta que contiene archivos de metadatos.
        attributes_df: DataFrame con atributos de los vehículos.
        make_mapping: Diccionario de mapeo de IDs a nombres de marcas.
        model_mapping: Diccionario de mapeo de IDs a nombres de modelos.
        type_mapping: Diccionario de mapeo de IDs a tipos de vehículos.
        dataset_df: DataFrame final con el dataset procesado.

    Example:
        >>> maker = DataFrameMaker("/path/to/CompCarsDataset")
        >>> maker.load_metadata()
        >>> dataset = maker.build_dataset()
        >>> maker.save_dataset("output.csv")

    Raises:
        CompCarsDatasetError: Para errores específicos del dataset.
        FileNotFoundError: Si no se encuentran archivos o carpetas requeridos.
    """

    def __init__(self, base_path: Union[str, Path]) -> None:
        """
        Inicializa el generador de dataset.

        Args:
            base_path: Ruta base del dataset CompCars.

        Raises:
            CompCarsDatasetError: Si la ruta base no existe o es inválida.
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise CompCarsDatasetError(f"La ruta base no existe: {self.base_path}")

        # Configuración de rutas
        self.images_folder = self.base_path / 'image'
        self.labels_folder = self.base_path / 'label'
        self.misc_folder = self.base_path / 'misc'

        # Validación de carpetas requeridas
        self._validate_folder_structure()

        # Inicialización de atributos
        self.attributes_df: Optional[pd.DataFrame] = None
        self.make_mapping: Optional[Dict[str, str]] = None
        self.model_mapping: Optional[Dict[str, str]] = None
        self.type_mapping: Optional[Dict[str, str]] = None
        self.dataset_df: Optional[pd.DataFrame] = None

        logging.info(f"Inicializado DataFrameMaker para: {self.base_path}")

    def _validate_folder_structure(self) -> None:
        """
        Valida que existan las carpetas requeridas del dataset.

        Raises:
            CompCarsDatasetError: Si faltan carpetas requeridas.
        """
        required_folders = [self.images_folder, self.labels_folder, self.misc_folder]
        missing_folders = [folder for folder in required_folders if not folder.exists()]

        if missing_folders:
            missing_names = [folder.name for folder in missing_folders]
            raise CompCarsDatasetError(
                f"Carpetas faltantes en el dataset: {missing_names}"
            )

    def _load_metadata(self) -> None:
        """
        Carga los metadatos necesarios para generar el dataset.

        Este método carga los archivos de atributos y mapeos de nombres
        desde la carpeta 'misc' del dataset CompCars.

        Raises:
            CompCarsDatasetError: Si hay errores cargando los metadatos.
            FileNotFoundError: Si faltan archivos de metadatos.
        """
        try:
            logging.info("Cargando metadatos del dataset...")

            # Carga de atributos de vehículos
            attributes_path = self.misc_folder / ATTRIBUTES_FILE
            if not attributes_path.exists():
                raise FileNotFoundError(f"Archivo de atributos no encontrado: {attributes_path}")

            self.attributes_df = pd.read_csv(attributes_path, sep=' ')
            self.attributes_df['model_id'] = self.attributes_df['model_id'].astype(str)

            # Carga de archivos .mat
            self._load_mat_mappings()

            logging.info("Metadatos cargados exitosamente")

        except Exception as e:
            raise CompCarsDatasetError(f"Error cargando metadatos: {e}") from e

    def _load_mat_mappings(self) -> None:
        """Carga los mapeos desde archivos .mat."""
        # Carga archivos .mat
        make_model_path = self.misc_folder / MAKE_MODEL_FILE
        car_type_path = self.misc_folder / CAR_TYPE_FILE

        if not make_model_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {make_model_path}")
        if not car_type_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {car_type_path}")

        make_model_mat = scipy.io.loadmat(make_model_path)
        car_type_mat = scipy.io.loadmat(car_type_path)

        # Extracción de nombres
        make_names = self._extract_mat_strings(make_model_mat['make_names'])
        model_names = self._extract_mat_strings(make_model_mat['model_names'], allow_empty=True)
        car_types = [str(x[0]) for x in car_type_mat['types'][0]]

        # Creación de mapeos
        self.make_mapping = {str(i + 1): name for i, name in enumerate(make_names)}
        self.model_mapping = {str(i + 1): name for i, name in enumerate(model_names)}
        self.type_mapping = {str(i + 1): name for i, name in enumerate(car_types)}
        self.type_mapping['0'] = UNKNOWN_TYPE

    def _extract_mat_strings(self, mat_array: Any, allow_empty: bool = False) -> List[str]:
        """
        Extrae strings de arrays de MATLAB.

        Args:
            mat_array: Array de MATLAB con strings.
            allow_empty: Si permitir strings vacíos.

        Returns:
            Lista de strings extraídos.
        """
        result = []
        for x in mat_array:
            if len(x) > 0 and len(x[0]) > 0:
                result.append(str(x[0][0]))
            elif allow_empty:
                result.append('')
            else:
                result.append('Unknown')
        return result

    def _process_image_data(
        self,
        image_path: Path,
        viewpoint_filter: Set[int] = DEFAULT_VIEWPOINT_FILTER
    ) -> Optional[Dict[str, Any]]:
        """
        Procesa los datos de una imagen individual.

        Args:
            image_path: Ruta a la imagen a procesar.
            viewpoint_filter: Conjunto de viewpoints permitidos.

        Returns:
            Diccionario con datos de la imagen o None si se debe filtrar.

        Raises:
            CompCarsDatasetError: Si hay errores procesando la imagen.
        """
        try:
            # Construcción de ruta del archivo de etiqueta
            label_path = self._get_label_path(image_path)
            if not label_path.exists():
                logging.warning(f"Archivo de etiqueta no encontrado: {label_path}")
                return None

            # Extracción de información de la ruta
            relative_path = image_path.relative_to(self.images_folder)
            path_parts = relative_path.parts

            if len(path_parts) != 4:  # make_id/model_id/year/image_name
                logging.warning(f"Estructura de ruta inválida: {relative_path}")
                return None

            make_id, model_id, released_year, image_name = path_parts

            # Lectura de datos de etiqueta
            viewpoint, bbox = self._parse_label_file(label_path)

            # Filtro por viewpoint
            if viewpoint not in viewpoint_filter:
                return None

            return {
                'image_name': image_name.replace('.jpg', ''),
                'image_path': str(image_path),
                'make_id': make_id,
                'model_id': model_id,
                'released_year': released_year,
                'viewpoint': viewpoint,
                'bbox': bbox
            }

        except Exception as e:
            logging.warning(f"Error procesando imagen {image_path}: {e}")
            return None

    def _get_label_path(self, image_path: Path) -> Path:
        """Obtiene la ruta del archivo de etiqueta correspondiente."""
        return Path(
            str(image_path)
            .replace(str(self.images_folder), str(self.labels_folder))
            .replace('.jpg', '.txt')
        )

    def _parse_label_file(self, label_path: Path) -> tuple[int, List[float]]:
        """
        Parsea el archivo de etiqueta y extrae viewpoint y bounding box.

        Args:
            label_path: Ruta al archivo de etiqueta.

        Returns:
            Tupla con (viewpoint, bbox_coordinates).
        """
        with open(label_path, 'r', encoding='utf-8') as f:
            viewpoint = int(f.readline().strip())
            f.readline()  # Skip second line
            bb_coords = f.readline().strip()

        bbox = [float(coord) for coord in bb_coords.split()]
        return viewpoint, bbox

    def build_dataset(
        self,
        viewpoint_filter: Set[int] = DEFAULT_VIEWPOINT_FILTER,
        columns_to_keep: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Construye el dataset completo procesando todas las imágenes.

        Args:
            viewpoint_filter: Conjunto de viewpoints a incluir.
            columns_to_keep: Columnas a mantener en el dataset final.

        Returns:
            DataFrame con el dataset procesado.

        Raises:
            CompCarsDatasetError: Si hay errores construyendo el dataset.
        """
        if self.attributes_df is None:
            self._load_metadata()

        if columns_to_keep is None:
            columns_to_keep = DEFAULT_COLUMNS_TO_KEEP

        logging.info("Iniciando construcción del dataset...")

        # Procesamiento de todas las imágenes
        dataset_list = []
        total_images = 0

        for image_path in self.images_folder.rglob('*.jpg'):
            total_images += 1
            data = self._process_image_data(image_path, viewpoint_filter)
            if data:
                dataset_list.append(data)

            # Log de progreso cada 1000 imágenes
            if total_images % 1000 == 0:
                logging.info(f"Procesadas {total_images} imágenes...")

        if not dataset_list:
            raise CompCarsDatasetError("No se encontraron imágenes válidas en el dataset")

        # Creación del DataFrame
        self.dataset_df = pd.DataFrame(dataset_list)

        # Aplicación de mapeos y transformaciones
        self._apply_mappings()

        # Filtrado de columnas
        self.dataset_df = self.dataset_df[columns_to_keep]

        # Estadísticas finales
        self._log_dataset_stats()

        logging.info(f"Dataset construido exitosamente con {len(self.dataset_df)} registros")
        return self.dataset_df

    def _apply_mappings(self) -> None:
        """Aplica los mapeos de nombres a los IDs numéricos."""
        # Mapeo de viewpoints
        self.dataset_df['viewpoint'] = self.dataset_df['viewpoint'].map(VIEWPOINT_MAPPING)

        # Merge con atributos
        self.dataset_df = self.dataset_df.merge(
            self.attributes_df, on='model_id', how='left'
        )

        # Aplicación de mapeos de nombres
        self.dataset_df['make'] = self.dataset_df['make_id'].map(self.make_mapping)
        self.dataset_df['model'] = self.dataset_df['model_id'].map(self.model_mapping)
        self.dataset_df['type'] = (
            self.dataset_df['type']
            .astype(int)
            .astype(str)
            .map(self.type_mapping)
        )

    def _log_dataset_stats(self) -> None:
        """Registra estadísticas del dataset."""
        logging.info(f"Dataset final: {len(self.dataset_df)} imágenes")

        # Distribución de viewpoints
        viewpoint_counts = self.dataset_df['viewpoint'].value_counts().to_dict()
        logging.info(f"Distribución de viewpoints: {viewpoint_counts}")

        # Distribución de tipos de vehículos
        type_counts = self.dataset_df['type'].value_counts().head(5).to_dict()
        logging.info(f"Top 5 tipos de vehículos: {type_counts}")

    def save_dataset(self, output_path: Union[str, Path], **kwargs) -> None:
        """
        Guarda el dataset en un archivo CSV.

        Args:
            output_path: Ruta donde guardar el archivo CSV.
            **kwargs: Argumentos adicionales para pandas.to_csv().

        Raises:
            CompCarsDatasetError: Si no hay dataset para guardar.
        """
        if self.dataset_df is None:
            raise CompCarsDatasetError(
                "No hay dataset para guardar. Ejecute build_dataset() primero."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Parámetros por defecto para CSV
        csv_params = {'index': False, 'encoding': 'utf-8'}
        csv_params.update(kwargs)

        self.dataset_df.to_csv(output_path, **csv_params)
        logging.info(f"Dataset guardado en: {output_path}")

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen estadístico del dataset.

        Returns:
            Diccionario con estadísticas del dataset.

        Raises:
            CompCarsDatasetError: Si no hay dataset construido.
        """
        if self.dataset_df is None:
            raise CompCarsDatasetError(
                "No hay dataset construido. Ejecute build_dataset() primero."
            )

        return {
            'total_images': len(self.dataset_df),
            'viewpoint_distribution': self.dataset_df['viewpoint'].value_counts().to_dict(),
            'make_distribution': self.dataset_df['make'].value_counts().head(10).to_dict(),
            'type_distribution': self.dataset_df['type'].value_counts().to_dict(),
            'year_range': (
                self.dataset_df['released_year'].min(),
                self.dataset_df['released_year'].max()
            ),
            'columns': list(self.dataset_df.columns)
        }

    def __repr__(self) -> str:
        """Representación string para debugging."""
        status = "construido" if self.dataset_df is not None else "no construido"
        size = len(self.dataset_df) if self.dataset_df is not None else 0
        return f"DataFrameMaker(base_path={self.base_path}, dataset={status}, size={size})"


# Funciones de conveniencia
def create_compcars_dataset(
    base_path: Union[str, Path],
    output_path: Union[str, Path],
    viewpoint_filter: Set[int] = DEFAULT_VIEWPOINT_FILTER
) -> pd.DataFrame:
    """
    Función de conveniencia para crear un dataset CompCars completo.

    Args:
        base_path: Ruta base del dataset CompCars.
        output_path: Ruta donde guardar el CSV.
        viewpoint_filter: Viewpoints a incluir.

    Returns:
        DataFrame con el dataset procesado.
    """
    maker = DataFrameMaker(base_path)
    dataset = maker.build_dataset(viewpoint_filter=viewpoint_filter)
    maker.save_dataset(output_path)
    return dataset