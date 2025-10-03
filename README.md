# CompCars - Análisis de Vehículos con Deep Learning

[![Python 3.12.6](https://img.shields.io/badge/python-3.12.6-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción

Proyecto de investigación para clasificación de vehículos utilizando el dataset **CompCars** con técnicas de deep learning y visión computacional. Implementa modelos de visión multi-vista, análisis de embeddings, clustering y pipelines reproducibles para fine-tuning.

### Objetivos Principales

- **Fine-tuning** de modelos pre-entrenados (ResNet, ViT, **CLIP**, etc.)
- **Modelos multimodales** con CLIP (imagen + texto)
- **Análisis de embeddings** antes y después del fine-tuning  
- **Clustering y visualización** de representaciones aprendidas
- **Soporte multi-vista** (front/rear) de vehículos
- **Pipelines reproducibles** con configuración JSON

## Arquitectura del Proyecto

```
CompCars/
├── src/                          # Código fuente principal
│   ├── config/                   # Configuraciones
│   │   ├── TransformConfig.py   # Transformaciones de imágenes
│   │   └── __init__.py
│   ├── data/                     # Procesamiento de datos      
│   │   ├── DataFrameMaker.py    # Generación de dataset CSV
│   │   ├── MyDataset.py         # Dataset personalizado PyTorch
│   │   └── __init__.py
│   ├── models/                   # Modelos de ML
│   │   ├── Criterions.py        # Criterions para metric learning
│   │   ├── MyVisionModel.py     # Modelo multi-vista (visión pura)
│   │   ├── MyCLIPModel.py       # Modelo CLIP (visión + lenguaje)
│   │   └── __init__.py
│   ├── pipeline/                 # Pipelines de experimentación
│   │   ├── FineTuningPipeline.py      # Pipeline finetuning
│   │   ├── EmbeddingsPipeline.py      # Pipeline de análisis
│   │   └── __init__.py
│   └── utils/                    # Utilidades
│       ├── ClusteringAnalyzer.py # Análisis de clustering
│       ├── DimensionalityReducer.py # Reducción dimensional
│       ├── ClusterVisualizer.py  # Visualización
│       ├── JsonUtils.py         # Utilidades JSON
│       └── __init__.py
├── dataset/                      # Dataset CompCars
│   ├── image/                   # Imágenes por categorías
│   └── label/                   # Etiquetas y metadatos
├── experiments/                  # Configuraciones experimentales
├── results/                      # Resultados y modelos guardados
├── requirements.txt              # Dependencias completas
├── requirements-dev.txt          # Dependencias esenciales
├── dataset.csv                   # Dataset creado por DataFrameMaker.py
├── config.json                   # Configs de finetuning y estudio de embeddings
└── README.md                    # Este archivo
```

## Instalación y Configuración

### 1. Prerrequisitos

- Python 3.12.6+
- CUDA compatible GPU (recomendado)
- Git LFS para archivos grandes del dataset

### 2. Clonación del repositorio

```bash
git clone https://github.com/diegovega2001/Memoria.git
cd Memoria/CompCars
```

### 3. Entorno virtual

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno (macOS/Linux)
source .venv/bin/activate

# Activar entorno (Windows)
.venv\\Scripts\\activate
```

### 4. Instalación de dependencias

```bash
# Instalación completa
pip install -r requirements.txt

# O instalación solo dependencias esenciales
pip install -r requirements-dev.txt
```

### 5. Instalación del proyecto

```bash
pip install -e .
```

## 💡 Uso Rápido

### Importaciones básicas

```python
# Modelos de visión pura
from src.models import MultiViewVisionModel, create_vision_model

# Modelos CLIP (visión + lenguaje)
from src.models import MultiViewCLIPModel, create_clip_model

# Pipelines
from src.pipeline import FineTuningPipeline, CLIPFineTuningPipeline

# Utilidades
from src.config import TransformConfig
from src.data import create_car_dataset
from src.utils import ClusteringAnalyzer
```

### 1. Crear dataset

```python
from src.data import create_car_dataset
from src.config import create_standard_transform

# Configurar transformaciones
transform = create_standard_transform(
    resize=(224, 224),
    normalize=True,
    augment=False
)

# Crear dataset
dataset = create_car_dataset(
    csv_path="dataset.csv",
    root_dir="dataset/",
    transform=transform,
    views=['front', 'rear'],
    model_type='vision'
)
```

### 2. Fine-tuning de modelo

```python
from src.pipeline import create_finetuning_pipeline

# Crear pipeline
pipeline = create_finetuning_pipeline(
    model_name="resnet50",
    dataset=dataset,
    batch_size=32,
    learning_rate=1e-4,
    epochs=10
)

# Ejecutar fine-tuning
pipeline.run_finetuning()
```

### 3. Análisis de embeddings

```python
from src.pipeline import create_embeddings_pipeline

# Pipeline de análisis
embeddings_pipeline = create_embeddings_pipeline(
    model_path="results/model.pth",
    dataset=dataset
)

# Extraer y analizar embeddings
embeddings_pipeline.extract_embeddings()
embeddings_pipeline.analyze_embeddings()
```

## Flujo de Trabajo Típico

### Opción 1: Modelos de Visión Pura

```python
# 1. Preparar datos
dataset = create_car_dataset(...)

# 2. Extraer embeddings baseline
pipeline = create_finetuning_pipeline(...)
pipeline.extract_baseline_embeddings()

# 3. Fine-tuning
pipeline.run_finetuning()

# 4. Extraer embeddings fine-tuned
pipeline.extract_finetuned_embeddings()

# 5. Análisis comparativo
embeddings_pipeline = create_embeddings_pipeline(...)
embeddings_pipeline.analyze_embeddings()
```

### Opción 2: Modelos CLIP (Visión + Lenguaje) 🆕

```python
from src.pipeline import create_clip_finetuning_pipeline

# 1. Configurar pipeline CLIP
config = {
    'model_name': 'clip-vit-base-patch32',
    'model_type': 'both',  # Imágenes + texto
    'description_include': 'all',  # Usar toda la info textual
    'objective': 'metric_learning',
    # ... más config
}

# 2. Crear pipeline
pipeline = create_clip_finetuning_pipeline(config, df)

# 3. Ejecutar fine-tuning por fases
results = pipeline.run_full_pipeline(
    phases=['text', 'projection', 'vision', 'projection_refine'],
    embedding_mode='joint'  # imagen + texto
)

# 4. Guardar resultados
pipeline.save_results("results/clip_experiments")
```

Ver [docs/CLIP_USAGE.md](docs/CLIP_USAGE.md) para documentación completa de CLIP.

## Configuración

### TransformConfig

```python
from src.config import TransformConfig

config = TransformConfig(
    grayscale=False,           # Mantener colores
    resize=(224, 224),         # Tamaño ImageNet
    normalize=True,            # Normalización ImageNet
    use_bbox=True             # Usar bounding boxes
)
```

### Configuración de clustering

```python
from src.utils import ClusteringAnalyzer

analyzer = ClusteringAnalyzer(
    algorithms=['dbscan', 'hdbscan', 'optics'],
    optimize_hyperparameters=True,
    n_jobs=-1
)
```

## Características Principales

### **Reproducibilidad**
- Seeds controladas en todos los componentes
- Configuraciones JSON persistentes
- Logging detallado de experimentos

### **Modularidad**
- Pipelines independientes y reutilizables
- Componentes intercambiables
- APIs consistentes

### **Análisis Completo**
- Clustering con múltiples algoritmos
- Reducción dimensional (PCA, t-SNE, UMAP)
- Métricas de evaluación detalladas

### **Multi-Vista**
- Soporte nativo para múltiples viewpoints
- Fusión de características multi-vista
- Análisis comparativo por vista

### **Persistencia**
- Guardado automático de resultados
- Versionado de modelos y embeddings
- Visualizaciones exportables

## Dataset CompCars

El proyecto utiliza el dataset **CompCars** que contiene:

- **163 marcas de vehículos**
- **1,716 modelos diferentes**
- **~214,000 imágenes**
- **Múltiples viewpoints** (front, rear, side)
- **Bounding boxes** para cada vehículo
- **Metadatos** (año, tipo, modelo)

### Estructura del CSV generado:

```csv
image_name,image_path,released_year,viewpoint,bbox,make,model,type
826a5fd082682c,dataset/image/135/947/unknown/826a5fd082682c.jpg,unknown,rear,"[96.0, 53.0, 817.0, 596.0]",Saab,SAAB 9X,Unknown
```

## Testing

```bash
# Ejecutar tests básicos
python -m pytest tests/

# Tests con coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Logging

El proyecto incluye logging configurado:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Los mensajes aparecerán automáticamente
```

## Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Autor

**Diego Vega** - [diegovega2001](https://github.com/diegovega2001)