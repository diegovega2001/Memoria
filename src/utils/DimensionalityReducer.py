import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
import umap
import optuna
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DimensionalityReducer:
    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        seed: int = 3,
        optimizer_trials: int = 33,
        available_methods: list = ['pca', 'umap']
    ):
        self.embeddings = embeddings.cpu().numpy() 
        self.labels = labels.numpy()
        self.seed = seed
        self.optimizer_trials = optimizer_trials
        
        self.available_methods = available_methods
        self.results = {}
        self.best_params = {}
        
    def _get_pca_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for PCA optimization."""

        return {
            'n_components': (0.9, 1.0)
        }
    
    def _get_tsne_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for t-SNE optimization."""

        return {
            'n_components': (2, 3),
            'perplexity': (5, 30),
            'learning_rate': (10, 1000),
            'max_iter': (750, 1250)
        }
    
    def _get_umap_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for UMAP optimization."""

        return {
            'n_components': (2, 25),
            'n_neighbors': (15, 100),
            'min_dist': (0.0, 0.25),
            'learning_rate': (0.5, 1.5)
        }
    
    def _create_reducer(self, method: str, params: Dict[str, Any]):
        """Create reducer instance with given parameters."""

        if method == 'pca':
            return PCA(random_state=self.seed, **params)
        
        elif method == 'tsne':
            return TSNE(random_state=self.seed, **params)
        
        elif method == 'umap':
            return umap.UMAP(random_state=self.seed, **params)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def optimize_parameters(self, method: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific method using Optuna."""
        
        def objective(trial):
            try:
                if method == 'pca':
                    param_ranges = self._get_pca_params_range()
                    params = {
                        'n_components': trial.suggest_float('n_components', *param_ranges['n_components'])
                    }
                    
                elif method == 'tsne':
                    param_ranges = self._get_tsne_params_range()
                    params = {
                        'n_components': trial.suggest_int('n_components', *param_ranges['n_components']),
                        'perplexity': trial.suggest_int('perplexity', *param_ranges['perplexity']),
                        'learning_rate': trial.suggest_float('learning_rate', *param_ranges['learning_rate']),
                        'max_iter': trial.suggest_int('max_iter', *param_ranges['max_iter'])
                    }
                    
                elif method == 'umap':
                    param_ranges = self._get_umap_params_range()
                    params = {
                        'n_components': trial.suggest_int('n_components', *param_ranges['n_components']),
                        'n_neighbors': trial.suggest_int('n_neighbors', *param_ranges['n_neighbors']),
                        'min_dist': trial.suggest_float('min_dist', *param_ranges['min_dist']),
                        'learning_rate': trial.suggest_float('learning_rate', *param_ranges['learning_rate'])
                    }
                
                reducer = self._create_reducer(method, params)
                reduced_embeddings = reducer.fit_transform(self.embeddings)
                silhouette_val = silhouette_score(reduced_embeddings, self.labels)
                trust_val = trustworthiness(self.embeddings, reduced_embeddings, n_neighbors=10)
                combined_score = 0.9 * silhouette_val + 0.1 * trust_val


                return combined_score
                
            except Exception as e:
                return -1
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.optimizer_trials, show_progress_bar=False)
        
        return study.best_params
    
    def reduce(self, method: str, params: Dict[str, Any] = None) -> np.ndarray:
        """Apply dimensionality reduction with given or optimized parameters."""

        if params is None:
            params = self.optimize_parameters(method)
            self.best_params[method] = params
            print(f"Best params for {method}: {params}")
        
        reducer = self._create_reducer(method, params)
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        return reduced_embeddings
    
    def reduce_all(self, methods: List[str] = None, optimize: bool = True) -> Dict[str, np.ndarray]:
        """Apply all specified reduction methods."""

        if methods is None:
            methods = self.available_methods
        
        results = {}

        for method in methods:            
            if optimize:
                reduced_embeddings = self.reduce(method)
            else:
                default_params = self._get_default_params(method)
                reduced_embeddings = self.reduce(method, default_params)
            
            results[method] = reduced_embeddings            
            score = silhouette_score(reduced_embeddings, self.labels)
                    
        self.results = results

        return results
    
    def _get_default_params(self, method: str) -> Dict[str, Any]:
        """Get default parameters for each method."""

        defaults = {
            'pca': {'n_components': min(50, self.embeddings.shape[1] - 1)},
            'tsne': {'n_components': 2, 'perplexity': 30, 'learning_rate': 200, 'max_iter': 500},
            'umap': {'n_components': 2, 'n_neighbors': 15, 'min_dist': 0.1, 'learning_rate': 1.0}
        }
        return defaults.get(method, {})
    
    def compare_methods(self, methods: List[str] = None) -> Dict[str, float]:
        """Compare different reduction methods using silhouette score."""

        if not self.results:
            self.reduce_all(methods)
        
        scores = {}
        print("\nComparison Results:")
        print("-" * 40)
        
        for method, embeddings in self.results.items():
            score = silhouette_score(embeddings, self.labels)
            scores[method] = score
            print(f"{method.upper():>8}: {score:.4f}")
        
        best_method = max(scores.keys(), key=lambda k: scores[k])
                
        return scores
    
    def get_best_result(self) -> Tuple[str, np.ndarray]:
        """Get the best reduction result based on silhouette score."""
        if not self.results:
            raise ValueError("No results available. Run reduce_all() first.")
        
        scores = self.compare_methods()
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        return best_method, self.results[best_method]
    
    def evaluate_reduction(self, reduced_embeddings: np.ndarray, method_name: str = "") -> Dict[str, float]:
        """Evaluate quality of dimensionality reduction."""
        metrics = {}
        
        metrics['silhouette_score'] = silhouette_score(reduced_embeddings, self.labels)
        
        unique_labels = np.unique(self.labels)
        within_cluster_ss = 0
        for label in unique_labels:
            cluster_points = reduced_embeddings[self.labels == label]
            if len(cluster_points) > 1:
                centroid = np.mean(cluster_points, axis=0)
                within_cluster_ss += np.sum((cluster_points - centroid) ** 2)
        metrics['within_cluster_ss'] = within_cluster_ss
        
        if method_name:
            print(f"\nEvaluation for {method_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        return metrics