DEFAULT_SEED = 3 # Lucky number
DEFAULT_N_JOBS = -1

# Constantes para normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]

# Configuración de augmentación agresiva
DEFAULT_USE_AUGMENT = False
DEFAULT_COLOR_JITTER_BRIGHTNESS = 0.3
DEFAULT_COLOR_JITTER_CONTRAST = 0.3
DEFAULT_COLOR_JITTER_SATURATION = 0.2
DEFAULT_COLOR_JITTER_HUE = 0.05
DEFAULT_ROTATION_DEGREES = 5
DEFAULT_RANDOM_ERASING_P = 0.1

# Constantes de inicialización de la transformación
DEFAULT_GRAYSCALE = True
DEFAULT_RESIZE = (224, 224)
DEFAULT_NORMALIZE = True
DEFAULT_USE_BBOX = True
DEFAULT_AUGMENT = True

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

# Constantes del dataset
DEFAULT_VIEWS = ['front']
DEFAULT_SEED = 3
DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS = 6 
DEFAULT_P = 8  # Número de clases por batch para contrastive learning
DEFAULT_K = 4  # Número de muestras por clase en cada batch
MODEL_TYPES = {'vision', 'textual', 'both'}
DEFAULT_MODEL_TYPE = 'vision'
DESCRIPTION_OPTIONS = {'', 'released_year', 'type', 'all'}
DEFAULT_DESCRIPTION_INCLUDE = ''
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_VERBOSE = True
UNKNOWN_VALUES = {'unknown', 'Unknown', '', None}

# Configuración fija para muestreo adaptativo
ADAPTIVE_RATIOS = {
    'abundant': {'train': 0.8, 'val': 0.1, 'test': 0.1},
    'few_shot': {'train': 0.7, 'val': 0.15, 'test': 0.15},
    'single_shot': {'train': 0.0, 'val': 0.0, 'test': 1.0}  
}

# Criterions
DEFAULT_OUTPUT_EMBEDDING_DIM = 512
DEFAULT_TRIPLET_MARGIN = 1.0
DEFAULT_CONTRASTIVE_MARGIN = 1.0
DEFAULT_ARCFACE_SCALE = 16.0
DEFAULT_ARCFACE_MARGIN = 0.25

# MyVisionModel
DEFAULT_MODEL_NAME = 'vit_b_16'
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_WEIGHTS_FILENAME = 'vision_model.pth'
DEFAULT_WEIGHTS = 'IMAGENET1K_V1'
DEFAULT_OBJECTIVE = 'metric_learning'
DEFAULT_PIN_MEMORY = False

# Dimensionality Reducer
DEFAULT_DIMENSIONALITY_REDUCER_OPTUNA_TRIALS = 20
DEFAULT_REDUCER_AVAILABLE_METHODS = ['pca', 'tsne', 'umap']
DEFAULT_USE_INCREMENTAL_PCA = True
DEFAULT_INCREMENTAL_BATCH_SIZE = 1000

# Cluster Analyzer
DEFAULT_CLUSTERING_OPTUNA_TRIALS = 120
DEFAULT_CLUSTERING_AVAILABLE_METHODS = ['dbscan', 'hdbscan', 'agglomerative', 'optics']

# Finetuning
DEFAULT_FINETUNE_CRITERION = 'ArcFaceLoss'
DEFAULT_FINETUNE_OPTIMIZER_TYPE ='AdamW'
DEFAULT_BACKBONE_LR = 1e-3
DEFAULT_HEAD_LR = 1e-2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_USE_SCHEDULER = True
DEFAULT_USE_EARLY_STOPPING = True
DEFAULT_PATIENCE = 5
DEFAULT_FINETUNE_EPOCHS = 50