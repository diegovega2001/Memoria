import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    adjusted_rand_score, 
    normalized_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import optuna
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ClusteringAnalyzer:
    def __init__(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        seed: int = 3,
        optimizer_trials: int = 33,
        available_methods: list = ['kmeans', 'dbscan', 'agglomerative', 'gaussian_mixture']
    ):
        self.embeddings = embeddings
        self.true_labels = true_labels
        self.seed = seed
        self.optimizer_trials = optimizer_trials
        
        scaler = StandardScaler()
        self.embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        self.n_true_clusters = len(np.unique(self.true_labels))
        self.available_methods = available_methods
        self.results = {}
        self.best_params = {}
        self.cluster_labels = {}
        
    def _get_kmeans_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for K-Means optimization."""

        return {
            'n_clusters': (self.n_true_clusters * 0.5, self.n_true_clusters),
            'init': ['k-means++'],
            'n_init': (10, 20)
        }
    
    def _get_dbscan_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for DBSCAN optimization."""

        k = 4
        neighbors = NearestNeighbors(n_neighbors=k).fit(self.embeddings_scaled)
        distances, indices = neighbors.kneighbors(self.embeddings_scaled)
        distances = np.sort(distances[:, k-1], axis=0)

        eps_min = np.percentile(distances, 15)
        eps_max = np.percentile(distances, 85)
        
        return {
            'eps': (eps_min, eps_max),
            'min_samples': (2, 10)
        }
    
    def _get_agglomerative_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for Agglomerative Clustering optimization."""

        return {
            'n_clusters': (self.n_true_clusters * 0.5, self.n_true_clusters),
            'linkage': ['ward', 'complete', 'average', 'single']
        }
    
    def _get_gaussian_mixture_params_range(self) -> Dict[str, Tuple]:
        """Define parameter ranges for gaussian Gaussian Mixture Clustering optimization."""

        return {
            'n_components': (self.n_true_clusters * 0.5, self.n_true_clusters),
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'init_params': ['kmeans']
        }

    
    def _create_clusterer(self, method: str, params: Dict[str, Any]):
        """Create clustering instance with given parameters."""

        if method == 'kmeans':
            return KMeans(random_state=self.seed, **params)
        
        elif method == 'dbscan':
            return DBSCAN(**params)
        
        elif method == 'agglomerative':
            return AgglomerativeClustering(**params)
        
        elif method == 'gaussian_mixture':
            return GaussianMixture(random_state=self.seed, **params)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def optimize_parameters(self, method: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific clustering method using Optuna."""
        
        def objective(trial):
            try:
                if method == 'kmeans':
                    param_ranges = self._get_kmeans_params_range()
                    params = {
                        'n_clusters': trial.suggest_int('n_clusters', *param_ranges['n_clusters']),
                        'init': trial.suggest_categorical('init', param_ranges['init']),
                        'n_init': trial.suggest_int('n_init', *param_ranges['n_init'])
                    }
                    
                elif method == 'dbscan':
                    param_ranges = self._get_dbscan_params_range()
                    params = {
                        'eps': trial.suggest_float('eps', *param_ranges['eps']),
                        'min_samples': trial.suggest_int('min_samples', *param_ranges['min_samples'])
                    }
                    
                elif method == 'agglomerative':
                    param_ranges = self._get_agglomerative_params_range()
                    params = {
                        'n_clusters': trial.suggest_int('n_clusters', *param_ranges['n_clusters']),
                        'linkage': trial.suggest_categorical('linkage', param_ranges['linkage'])
                    }
                    
                elif method == 'gaussian_mixture':
                    param_ranges = self._get_gaussian_mixture_params_range()
                    params = {
                        'n_components': trial.suggest_int('n_components', *param_ranges['n_components']),
                        'covariance_type': trial.suggest_categorical('covariance_type', param_ranges['covariance_type']),
                        'init_params': trial.suggest_categorical('init_params', param_ranges['init_params'])
                    }
                
                clusterer = self._create_clusterer(method, params)
                cluster_labels = clusterer.fit_predict(self.embeddings_scaled)
                
                n_clusters = len(np.unique(cluster_labels))
                if n_clusters < 2 or (method == 'dbscan' and n_clusters == 1):  
                    return -1
                
                silhouette = silhouette_score(self.embeddings_scaled, cluster_labels)
                ari = adjusted_rand_score(self.true_labels, cluster_labels)
                nmi = normalized_mutual_info_score(self.true_labels, cluster_labels)

                combined_score = 0.2 * silhouette + 0.5 * ari + 0.3 * nmi

                return combined_score
                
            except Exception as e:
                return -1
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.optimizer_trials, show_progress_bar=False)
        
        return study.best_params
    
    def cluster(self, method: str, params: Dict[str, Any] = None) -> np.ndarray:
        """Apply clustering with given or optimized parameters."""

        if params is None:
            print(f"Optimizing parameters for {method.upper()}...")
            params = self.optimize_parameters(method)
            self.best_params[method] = params
            print(f"Best params for {method}: {params}")
        
        clusterer = self._create_clusterer(method, params)
        cluster_labels = clusterer.fit_predict(self.embeddings_scaled)
        
        return cluster_labels
    
    def cluster_all(self, methods: List[str] = None, optimize: bool = True) -> Dict[str, np.ndarray]:
        """Apply all specified clustering methods."""

        if methods is None:
            methods = self.available_methods
        
        results = {}

        for method in methods:
            print(f"\nProcessing {method.upper()}...")
            
            if optimize:
                cluster_labels = self.cluster(method)
            else:
                default_params = self._get_default_params(method)
                cluster_labels = self.cluster(method, default_params)
            
            results[method] = cluster_labels
            self.cluster_labels[method] = cluster_labels
            
            metrics = self.evaluate_clustering(cluster_labels, method)
            print(f"{method.upper()} - Clusters: {len(np.unique(cluster_labels))}, "
                  f"Silhouette: {metrics['silhouette_score']:.4f}, "
                  f"ARI: {metrics['adjusted_rand_score']:.4f}")
        
        self.results = results
        return results
    
    def _get_default_params(self, method: str) -> Dict[str, Any]:
        """Get default parameters for each clustering method."""

        defaults = {
            'kmeans': {'n_clusters': self.n_true_clusters, 'init': 'k-means++', 'n_init': 10},
            'dbscan': {'eps': 0.5, 'min_samples': 5},
            'agglomerative': {'n_clusters': self.n_true_clusters, 'linkage': 'ward'},
            'gaussian_mixture': {'n_components': self.n_true_clusters, 'covariance_type': 'full', 'init_params': 'kmeans'}
        }

        return defaults.get(method, {})
    
    def evaluate_clustering(self, cluster_labels: np.ndarray, method_name: str = "") -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics."""

        metrics = {}
        
        valid_mask = cluster_labels != -1

        if np.sum(valid_mask) < 2: 
            return {
                'silhouette_score': -1,
                'adjusted_rand_score': -1,
                'normalized_mutual_info': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': float('inf'),
                'n_clusters': 1,
                'n_noise_points': np.sum(cluster_labels == -1)
            }

        valid_embeddings = self.embeddings_scaled[valid_mask]
        valid_cluster_labels = cluster_labels[valid_mask]
        valid_true_labels = self.true_labels[valid_mask]
        
        if len(np.unique(valid_cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(valid_embeddings, valid_cluster_labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(valid_embeddings, valid_cluster_labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(valid_embeddings, valid_cluster_labels)
        else:
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = -1
            metrics['davies_bouldin_score'] = float('inf')
        
        metrics['adjusted_rand_score'] = adjusted_rand_score(valid_true_labels, valid_cluster_labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(valid_true_labels, valid_cluster_labels)
        
        metrics['n_clusters'] = len(np.unique(valid_cluster_labels))
        metrics['n_noise_points'] = np.sum(cluster_labels == -1)
        
        return metrics
    
    def compare_methods(self, methods: List[str] = None) -> pd.DataFrame:
        """Compare different clustering methods using multiple metrics."""

        if not self.results:
            self.cluster_all(methods)
        
        comparison_data = []
        
        print("\nDetailed Comparison Results:")
        print("=" * 80)
        
        for method, cluster_labels in self.results.items():
            metrics = self.evaluate_clustering(cluster_labels, method)
            
            row_data = {'method': method}
            row_data.update(metrics)
            comparison_data.append(row_data)
            
            print(f"\n{method.upper()}:")
            print(f"  Clusters: {metrics['n_clusters']}")
            if method == 'dbscan':
                print(f"  Noise points: {metrics['n_noise_points']}")
            print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"  Adjusted Rand Index: {metrics['adjusted_rand_score']:.4f}")
            print(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
            print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
            print(f"  Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"\n{'BEST METHODS':=^80}")
        
        metrics_to_rank = ['silhouette_score', 'adjusted_rand_score', 'normalized_mutual_info', 'calinski_harabasz_score']
        for metric in metrics_to_rank:
            if metric == 'davies_bouldin_score':
                best_idx = comparison_df[metric].idxmin()
            else:  
                best_idx = comparison_df[metric].idxmax()
            best_method = comparison_df.loc[best_idx, 'method']
            best_value = comparison_df.loc[best_idx, metric]
            print(f"{metric:25}: {best_method.upper():15} ({best_value:.4f})")
        
        return comparison_df
    
    def get_best_result(self, metric: str = 'adjusted_rand_score') -> Tuple[str, np.ndarray]:
        """Get the best clustering result based on specified metric."""

        if not self.results:
            raise ValueError("No results available. Run cluster_all() first.")
        
        best_score = -float('inf') if metric != 'davies_bouldin_score' else float('inf')
        best_method = None
        
        for method, cluster_labels in self.results.items():
            metrics = self.evaluate_clustering(cluster_labels)
            score = metrics[metric]
            
            if metric == 'davies_bouldin_score':  
                if score < best_score:
                    best_score = score
                    best_method = method
            else:  # Higher is better
                if score > best_score:
                    best_score = score
                    best_method = method
        
        return best_method, self.results[best_method]
    
    def get_cluster_summary(self, method: str) -> pd.DataFrame:
        """Get summary statistics for each cluster in a specific method."""
        if method not in self.results:
            raise ValueError(f"Method {method} not found. Available methods: {list(self.results.keys())}")
        
        cluster_labels = self.results[method]
        summary_data = []
        
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_true_labels = self.true_labels[mask]
            
            true_label_counts = pd.Series(cluster_true_labels).value_counts()
            most_common_true_label = true_label_counts.index[0] if len(true_label_counts) > 0 else -1
            
            summary_data.append({
                'cluster_id': cluster_id,
                'size': np.sum(mask),
                'most_common_true_label': most_common_true_label,
                'purity': true_label_counts.iloc[0] / np.sum(mask) if len(true_label_counts) > 0 else 0,
                'n_unique_true_labels': len(true_label_counts)
            })
        
        return pd.DataFrame(summary_data).sort_values('cluster_id')