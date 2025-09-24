import torch
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
import optuna
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
import gc
from numba import jit
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


@jit(nopython=True)
def fast_euclidean_distance(X, Y):
    """Fast euclidean distance computation using numba."""
    return np.sqrt(np.sum((X - Y) ** 2))


class DimensionalityReducer:
    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        seed: int = 3,
        optimizer_trials: int = 50,
        available_methods: list = ['pca', 'umap'],
        use_incremental: bool = True,
        n_jobs: int = -1,
        batch_size: int = 1000
    ):
        self.embeddings = embeddings.cpu().numpy()
        self.embeddings = self.embeddings.astype(np.float32)
        self.labels = labels.numpy()

        self.scaler = StandardScaler()
        self.embeddings = self.scaler.fit_transform(self.embeddings)

        self.seed = seed
        self.optimizer_trials = optimizer_trials
        self.available_methods = available_methods
        self.use_incremental = use_incremental
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count() -1

        self.results = {}
        self.best_params = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"Initialized with {self.embeddings.shape[0]} samples, {self.embeddings.shape[1]} features")
        logging.info(f"Using {self.n_jobs} CPU cores")        

    def _get_default_params(self, method: str) -> Dict[str, Any]:
        """Get default parameters optimized for clustering performance."""
        max_components = min(100, self.embeddings.shape[1] - 1) 
        
        defaults = {
            'pca': {'n_components': min(50, max_components)}, 
            'tsne': {
                'n_components': 2,
                'perplexity': min(30, self.embeddings.shape[0] // 4),
                'learning_rate': 200,
                'max_iter': 1000,  
                'early_exaggeration': 12  
            },
            'umap': {
                'n_components': 20,  
                'n_neighbors': min(30, self.embeddings.shape[0] // 20),  
                'min_dist': 0.0,  
                'learning_rate': 1.0,
                'metric': 'cosine' 
            }
        }
        return defaults.get(method, {})
    
    def _get_pca_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for PCA optimization - higher components for clustering."""
        n_samples = self.embeddings.shape[0]
        n_features = self.embeddings.shape[-1]
        max_possible = min(n_features, n_samples - 1)
        
        min_components = max(10, min(30, max_possible // 4))  
        max_components = min(max_possible, max(100, n_features // 2))  
        
        return {'n_components': (min_components, max_components)}
    
    def _get_tsne_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for t-SNE optimization."""
        n_samples = self.embeddings.shape[0]
        perplexity_min = max(5, min(15, n_samples // 150)) 
        perplexity_max = min(100, n_samples // 3) 
        return {
            'n_components': (2, 3),
            'perplexity': (perplexity_min, perplexity_max),
            'learning_rate': (100, 1000),  
            'max_iter': (1000, 2000), 
            'early_exaggeration': (6.0, 20.0) 
        }
    
    def _get_umap_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for UMAP optimization - clustering focused."""
        n_samples = self.embeddings.shape[0]
        neighbors_min = max(5, min(15, n_samples // 150))  
        neighbors_max = min(200, n_samples // 5)  
        return {
            'n_components': (2, min(100, self.embeddings.shape[1] // 2)), 
            'n_neighbors': (neighbors_min, neighbors_max),
            'min_dist': (0.0, 0.5),  
            'learning_rate': (0.1, 3.0),  
            'metric': ['euclidean', 'cosine', 'manhattan']  
        }
    
    def _create_reducer(self, method: str, params: Dict[str, Any]):
        """Create reducer instance with given parameters."""
        if method == 'pca':
            if self.use_incremental and self.embeddings.shape[0] > 10000:
                return IncrementalPCA(
                    n_components=params['n_components'],
                    batch_size=min(self.batch_size, self.embeddings.shape[0] // 10)
                )
            else:
                return PCA(random_state=self.seed,
                        **params)
        elif method == 'tsne':
            return TSNE(
                random_state=self.seed, 
                n_jobs=self.n_jobs,
                **params)
        elif method == 'umap':
            return umap.UMAP(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                low_memory=True,
                **params)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _optimize_parameters(self, method: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific method using Optuna."""
        def objective(trial):
            try:
                if method == 'pca':
                    param_ranges = self._get_pca_params_range()
                    params = {
                        'n_components': trial.suggest_int('n_components', *param_ranges['n_components'])
                    }  
                elif method == 'tsne':
                    param_ranges = self._get_tsne_params_range()
                    params = {
                        'n_components': trial.suggest_int('n_components', *param_ranges['n_components']),
                        'perplexity': trial.suggest_int('perplexity', *param_ranges['perplexity']),
                        'learning_rate': trial.suggest_float('learning_rate', *param_ranges['learning_rate']),
                        'max_iter': trial.suggest_int('max_iter', *param_ranges['max_iter']),
                        'early_exaggeration': trial.suggest_float('early_exaggeration', *param_ranges['early_exaggeration'])
                    } 
                elif method == 'umap':
                    param_ranges = self._get_umap_params_range()
                    params = {
                        'n_components': trial.suggest_int('n_components', *param_ranges['n_components']),
                        'n_neighbors': trial.suggest_int('n_neighbors', *param_ranges['n_neighbors']),
                        'min_dist': trial.suggest_float('min_dist', *param_ranges['min_dist']),
                        'learning_rate': trial.suggest_float('learning_rate', *param_ranges['learning_rate']),
                        'metric': trial.suggest_categorical('metric', param_ranges['metric'])
                    }
                reducer = self._create_reducer(method, params)
                reduced_embeddings = reducer.fit_transform(self.embeddings) 
                silhouette_val = silhouette_score(reduced_embeddings, self.labels)
                trust_val = trustworthiness(self.embeddings, reduced_embeddings, n_neighbors=10)
                silhouette_norm = (silhouette_val + 1) / 2
                combined_score = 0.6 * silhouette_norm + 0.4 * trust_val
                
                del reducer, reduced_embeddings
                gc.collect()
                return combined_score
            except Exception as e:
                logging.warning(f"Trial failed for {method}: {e}")
                return -1.0
            
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=min(10, self.optimizer_trials // 10),
            multivariate=True
        )
        study = optuna.create_study(
            direction='maximize', 
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner()
        )
        try:
            study.optimize(
                objective, 
                n_trials=self.optimizer_trials,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logging.info("Optimization interrupted by user")
        logging.info(f"Optimization completed for {method}. Best value: {study.best_value:.4f}")
        return study.best_params
    
    def reduce(self, method: str, params: Dict[str, Any] = None) -> np.ndarray:
        """Apply dimensionality reduction with given or optimized parameters."""
        if params is None:
            params = self._optimize_parameters(method)
            self.best_params[method] = params
            logging.info(f"Best params for {method}: {params}")
        
        reducer = self._create_reducer(method, params)
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        del reducer
        gc.collect()
        return reduced_embeddings
    
    def reduce_all(self, methods: List[str] = None, optimize: bool = True) -> Dict[str, np.ndarray]:
        """Apply all specified reduction methods."""
        if methods is None:
            methods = self.available_methods
        results = {}

        for method in methods:   
            logging.info(f"Processing {method.upper()}...")
            try:          
                if optimize:
                    reduced_embeddings = self.reduce(method)
                else:
                    default_params = self._get_default_params(method)
                    reduced_embeddings = self.reduce(method, default_params)
                
                results[method] = reduced_embeddings            
                score = silhouette_score(reduced_embeddings, self.labels)
                logging.info(f"{method.upper()} silhouette score: {score:.4f}")
                gc.collect()

            except Exception as e:
                logging.error(f"Failed to process {method}: {e}")
                continue

        self.results = results
        return results
    
    def compare_methods(self, methods: List[str] = None) -> Dict[str, float]:
        """Compare different reduction methods using silhouette score."""
        if not self.results:
            self.reduce_all(methods, optimize=True)
        
        scores = {}
        logging.info("\nComparison Results:")
        logging.info("-" * 40)
        
        for method, embeddings in self.results.items():
            score = silhouette_score(embeddings, self.labels)
            scores[method] = score
            logging.info(f"{method.upper():>8}: {score:.4f}")            
        return scores
    
    def get_best_result(self) -> Tuple[str, np.ndarray]:
        """Get the best reduction result based on silhouette score."""
        if not self.results:
            raise ValueError("No results available. Run reduce_all() first.")
        
        scores = self.compare_methods()
        best_method = max(scores.keys(), key=lambda k: scores[k])
        return best_method, self.results[best_method]