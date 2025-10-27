DEFAULT_SEED = 3 # Lucky number
DEFAULT_N_JOBS = -1

# Constantes para normalización ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]

# Configuración de augmentación 
DEFAULT_USE_AUGMENT = False
DEFAULT_COLOR_JITTER_BRIGHTNESS = 0.15
DEFAULT_COLOR_JITTER_CONTRAST = 0.15
DEFAULT_COLOR_JITTER_SATURATION = 0.1
DEFAULT_COLOR_JITTER_HUE = 0.02
DEFAULT_ROTATION_DEGREES = 3
DEFAULT_RANDOM_ERASING_P = 0.05

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
DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS = 7
DEFAULT_P = 16
DEFAULT_K = 4
MODEL_TYPES = {'vision', 'textual', 'both'}
DEFAULT_MODEL_TYPE = 'vision'
DESCRIPTION_OPTIONS = {'', 'basic', 'model', 'all'}
DEFAULT_DESCRIPTION_INCLUDE = ''
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0
DEFAULT_VERBOSE = True
UNKNOWN_VALUES = {'unknown', 'Unknown', '', None}

CLASS_GRANULARITY_OPTIONS = {'model', 'model+year'}
DEFAULT_CLASS_GRANULARITY = 'model+year'
DEFAULT_TEST_UNSEEN_ENABLED = False
DEFAULT_TEST_UNSEEN_COUNT = 400
TEST_UNSEEN_STRATEGY_OPTIONS = {'balanced', 'temporal_only', 'random'}
DEFAULT_TEST_UNSEEN_STRATEGY = 'balanced'
DEFAULT_TEMPORAL_RATIO = 0.25
DEFAULT_INTRA_MAKE_RATIO = 0.25
DEFAULT_INTER_MAKE_RATIO = 0.50

# Configuración fija para muestreo adaptativo
ADAPTIVE_RATIOS = {
    'abundant': {'train': 0.43, 'val': 0.29, 'test': 0.29},
    'few_shot': {'train': 0.5, 'val': 0.25, 'test': 0.25},
    'single_shot': {'train': 0.0, 'val': 0.0, 'test': 1.0}  
}

# Criterions
DEFAULT_OUTPUT_EMBEDDING_DIM = 512
DEFAULT_TRIPLET_MARGIN = 1.0
DEFAULT_CONTRASTIVE_MARGIN = 1.0
DEFAULT_ARCFACE_SCALE = 30.0
DEFAULT_ARCFACE_MARGIN = 0.50

# MyVisionModel
SUPPORTED_MODEL_ATTRIBUTES = {'fc', 'heads', 'classifier', 'head'}
MODEL_CONFIGS = {
    'resnet': {'feature_attr': 'fc', 'feature_key': 'in_features'},
    'densenet': {'feature_attr': 'classifier', 'feature_key': 'in_features'},
    'efficientnet': {'feature_attr': 'classifier', 'feature_key': 'in_features'},
    'vit': {'feature_attr': 'heads', 'feature_key': 'in_features'},
    'swin': {'feature_attr': 'head', 'feature_key': 'in_features'},
}
DEFAULT_MODEL_NAME = 'vit_b_32'
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_WEIGHTS_FILENAME = 'vision_model.pth'
DEFAULT_WEIGHTS = 'IMAGENET1K_V1'
DEFAULT_OBJECTIVE = 'metric_learning'
DEFAULT_PIN_MEMORY = True

# MyCLIPModel
DEFAULT_CLIP_MODEL_NAME = 'clip-vit-base-patch32'
CLIP_EMBEDDING_MODES = {'image', 'text', 'joint'}
DEFAULT_CLIP_EMBEDDING_MODE = 'joint'
CLIP_DEFAULT_FINETUNING_PHASES = {
    'phase 1': {
        'type': 'text',
        'lr': 1e-5,
        'epochs': 15,
        'early_stopping': None,
        'save_best': True,
        'warmup_steps': 200
    },
    'phase 2': {
        'type': 'projection',
        'lr': 1e-5,
        'epochs': 15,
        'early_stopping': None,
        'save_best': True,
        'warmup_steps': 200
    },
    'phase 3': {
        'type': 'vision',
        'lr': 1e-5,
        'epochs': 15,
        'early_stopping': None,
        'save_best': True,
        'warmup_steps': 200,
        'num_vision_layers': 1
    },
    'phase 4': {
        'type': 'projection_refine',
        'lr': 1e-5,
        'epochs': 15,
        'early_stopping': None,
        'save_best': True,
        'warmup_steps': 200
    }
}
# Configuraciones conocidas de modelos CLIP
CLIP_CONFIGS = {
    'clip-vit-base-patch32': {'model_name': 'openai/clip-vit-base-patch32'},
    'clip-vit-base-patch16': {'model_name': 'openai/clip-vit-base-patch16'},
    'clip-vit-large-patch14': {'model_name': 'openai/clip-vit-large-patch14'}
}

# Dimensionality Reducer
DEFAULT_DIMENSIONALITY_REDUCER_OPTUNA_TRIALS = 20
DEFAULT_REDUCER_AVAILABLE_METHODS = ['pca', 'tsne', 'umap']
DEFAULT_USE_INCREMENTAL_PCA = True
DEFAULT_INCREMENTAL_BATCH_SIZE = 1000

# Cluster Analyzer
DEFAULT_CLUSTERING_OPTUNA_TRIALS = 120
DEFAULT_CLUSTERING_AVAILABLE_METHODS = ['dbscan', 'hdbscan', 'agglomerative', 'optics']

# Vision Finetuning
DEFAULT_FINETUNE_CRITERION = 'ContrastiveLoss'
DEFAULT_FINETUNE_OPTIMIZER_TYPE ='AdamW'
DEFAULT_BACKBONE_LR = 1e-4
DEFAULT_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 5e-3
DEFAULT_USE_SCHEDULER = True
DEFAULT_SCHEDULER_TYPE = 'cosine_warmup'
DEFAULT_USE_EARLY_STOPPING = True
DEFAULT_PATIENCE = 10
DEFAULT_FINETUNE_EPOCHS = 50
DEFAULT_GRADIENT_CLIP_VALUE = 1.0

# AMP
DEFAULT_USE_AMP = True 

# Mensajes de error
ERROR_INVALID_MODEL_TYPE = "model_type debe ser uno de: {}"
ERROR_INVALID_DESCRIPTION = "description_include debe ser uno de: {}"
