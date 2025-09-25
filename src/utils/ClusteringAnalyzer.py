"""
Analizador de clustering optimizado para análisis de embeddings.

Este módulo proporciona métodos de clustering (DBSCAN, HDBSCAN, OPTICS, Agglomerative)
con optimización automática de hiperparámetros enfocada en análisis de embeddings.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import warnings
from typing import Any, Dict, List, Tuple, Union

import hdbscan
import numpy as np
import optuna
import pandas as pd
from numba import jit
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


@jit(nopython=True)
def fast_purity_calculation(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Cálculo rápido de pureza usando numba para optimización.
    
    Args:
        cluster_labels: Etiquetas de cluster asignadas
        true_labels: Etiquetas verdaderas
        
    Returns:
        Valor de pureza entre 0 y 1
    """
    cluster_labels = cluster_labels.astype(np.int32)
    true_labels = true_labels.astype(np.int32)
    
    unique_clusters = np.unique(cluster_labels)
    total_samples = len(cluster_labels)
    correct_assignments = 0
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Ignorar puntos de ruido
            continue
            
        mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[mask]

        if len(cluster_true_labels) > 0:
            # Encontrar la etiqueta más frecuente en este cluster
            unique_true = np.unique(cluster_true_labels)
            max_count = 0
            for val in unique_true:
                count = np.sum(cluster_true_labels == val)
                if count > max_count:
                    max_count = count
            correct_assignments += max_count

    return correct_assignments / total_samples


class ClusteringAnalyzer:
    """
    Analizador de clustering optimizado para análisis de embeddings.
    
    Proporciona métodos de clustering con optimización automática de hiperparámetros
    enfocada en la separabilidad y correspondencia con etiquetas verdaderas.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        seed: int = 3,
        optimizer_trials: int = 50,
        available_methods: List[str] = None,
        n_jobs: int = -1
    ):
        """
        Inicializa el analizador de clustering.
        
        Args:
            embeddings: Array de embeddings a analizar
            true_labels: Etiquetas verdaderas para evaluación
            seed: Semilla para reproducibilidad
            optimizer_trials: Número de trials para optimización
            available_methods: Lista de métodos disponibles
            n_jobs: Número de cores para paralelización
        """
        if available_methods is None:
            available_methods = ['dbscan', 'hdbscan', 'optics', 'agglomerative']
            
        # Conversión y preprocesamiento de datos
        if hasattr(embeddings, 'cpu'):  # Tensor de PyTorch
            self.embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            self.embeddings = embeddings.astype(np.float32)
            
        if hasattr(true_labels, 'cpu'):  # Tensor de PyTorch
            self.true_labels = true_labels.cpu().numpy().astype(np.int32)
        else:
            self.true_labels = true_labels.astype(np.int32)
        
        # Configuración
        self.seed = seed
        self.optimizer_trials = optimizer_trials
        self.available_methods = available_methods
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count() - 1
        
        # Información del dataset
        self.n_true_clusters = len(np.unique(self.true_labels))
        
        # Resultados y parámetros óptimos
        self.results: Dict[str, np.ndarray] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.cluster_labels: Dict[str, np.ndarray] = {}
        
        logging.info(f"Inicializado clustering con {len(self.embeddings)} muestras, {self.n_true_clusters} clusters verdaderos")
    

    
    def _get_dbscan_params_range(self) -> Dict[str, Tuple[float, int]]:
        """
        Define rangos de parámetros para optimización de DBSCAN.
        
        Utiliza análisis k-distance inteligente para determinar rangos óptimos
        de epsilon basados en la distribución de distancias del dataset.
        """
        n_samples, reduced_dims = self.embeddings.shape
        
        # Análisis k-distance optimizado
        k = max(4, min(10, n_samples // 120))  # Más adaptativos
        neighbors = NearestNeighbors(
            n_neighbors=k, 
            n_jobs=self.n_jobs, 
            algorithm='ball_tree'
        )
        neighbors.fit(self.embeddings)
        distances, _ = neighbors.kneighbors(self.embeddings)
        k_distances = np.sort(distances[:, k-1])
        
        # Rangos adaptativos según dimensionalidad
        if reduced_dims <= 3:
            eps_min = max(0.05, np.percentile(k_distances, 20))
            eps_max = min(4.0, np.percentile(k_distances, 88))
        elif reduced_dims <= 25:
            eps_min = max(0.3, np.percentile(k_distances, 30))
            eps_max = min(10.0, np.percentile(k_distances, 85))
        else:
            eps_min = max(0.8, np.percentile(k_distances, 35))
            eps_max = min(20.0, np.percentile(k_distances, 80))
        
        # Garantizar rango mínimo viable
        if eps_max - eps_min < 0.3:
            eps_max = eps_min + 1.5
        
        # min_samples optimizado para embeddings de visión
        min_samples_max = min(20, max(8, n_samples // 60))
        
        return {
            'eps': (eps_min, eps_max),
            'min_samples': (4, min_samples_max)  # Mínimo 4 para mayor robustez
        }
    
    def _get_hdbscan_params_range(self) -> Dict[str, Tuple[int, float]]:
        """
        Define rangos de parámetros para optimización de HDBSCAN.
        
        Calcula rangos adaptativos basados en el tamaño del dataset y
        características de los embeddings para maximizar calidad del clustering.
        """
        n_samples, reduced_dims = self.embeddings.shape
        
        # min_cluster_size optimizado para embeddings de visión
        min_cluster_min = max(6, n_samples // 180)  # Mínimo 0.55% datos
        min_cluster_max = min(150, n_samples // 6)   # Máximo 16.7%
        
        # min_samples como proporción inteligente de min_cluster_size
        min_samples_min = max(2, min_cluster_min // 3)
        min_samples_max = min(75, min_cluster_max // 2)
        
        # cluster_selection_epsilon para control de granularidad
        noise_threshold = self._estimate_noise_threshold()
        if reduced_dims <= 15:
            epsilon_min = 0.0
            epsilon_max = min(0.8, noise_threshold * 0.7)
        else:
            epsilon_min = 0.0
            epsilon_max = min(1.5, noise_threshold * 1.1)
        
        return {
            'min_cluster_size': (min_cluster_min, min_cluster_max),
            'min_samples': (min_samples_min, min_samples_max),
            'cluster_selection_epsilon': (epsilon_min, epsilon_max)
        }

    def _get_optics_params_range(self) -> Dict[str, Tuple[int, float]]:
        """
        Define rangos de parámetros para optimización de OPTICS.
        
        Configura rangos adaptativos optimizados para análisis de embeddings
        de visión computacional con enfoque en calidad de clustering.
        """
        n_samples, reduced_dims = self.embeddings.shape
        
        # min_samples adaptativo según dimensionalidad y tamaño
        if reduced_dims <= 10:
            min_samples_min, min_samples_max = 4, 12
        elif reduced_dims <= 50:
            min_samples_min, min_samples_max = 6, 18
        else:
            min_samples_min, min_samples_max = 8, 25
        
        # xi optimizado para separación de clusters en embeddings
        if n_samples < 500:
            xi_min, xi_max = 0.02, 0.12  # Más granular para datasets pequeños
        elif n_samples < 2000:
            xi_min, xi_max = 0.015, 0.09  # Balance medio
        else:
            xi_min, xi_max = 0.01, 0.07   # Más conservador para datasets grandes
        
        # min_cluster_size proporcional al dataset
        min_cluster_max = min(30, max(10, n_samples // 45))
        
        return {
            'min_samples': (min_samples_min, min_samples_max),
            'xi': (xi_min, xi_max),
            'min_cluster_size': (5, min_cluster_max)  # Mínimo 5 para estabilidad
        }

    def _get_agglomerative_params_range(self) -> Dict[str, Union[Tuple[float, float], List[str]]]:
        """
        Define rangos de parámetros para Clustering Aglomerativo.
        
        Utiliza análisis estadístico de distancias para calcular thresholds
        óptimos adaptados a las características del espacio de embeddings.
        """
        n_samples, reduced_dims = self.embeddings.shape
        
        # Muestreo inteligente para análisis de distancias
        sample_size = min(800, max(100, n_samples // 5))
        sample_idx = np.random.RandomState(self.seed).choice(
            n_samples, sample_size, replace=False
        )
        sample_data = self.embeddings[sample_idx]
        
        # Análisis de distribución de distancias
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(sample_data[:150])
        valid_distances = distances[distances > 0]
        
        # Cálculo de thresholds adaptativos
        percentile_25 = np.percentile(valid_distances, 25)
        percentile_75 = np.percentile(valid_distances, 75)
        mean_dist = np.mean(valid_distances)
        
        # Rangos optimizados según dimensionalidad
        if reduced_dims <= 10:
            threshold_min = max(2.0, percentile_25 * 0.8)
            threshold_max = min(25.0, percentile_75 * 2.2)
        elif reduced_dims <= 50:
            threshold_min = max(4.0, mean_dist * 0.6)
            threshold_max = min(40.0, mean_dist * 2.8)
        else:
            threshold_min = max(6.0, mean_dist * 0.7)
            threshold_max = min(60.0, mean_dist * 3.5)
        
        return {
            'distance_threshold': (threshold_min, threshold_max),
            'linkage': ['ward', 'complete', 'average']  # Incluir ward para embeddings normalizados
        }

    def _create_clusterer(self, method: str, params: Dict[str, Any]):
        """
        Crea instancia del algoritmo de clustering especificado.
        
        Args:
            method: Nombre del método ('dbscan', 'hdbscan', 'optics', 'agglomerative')
            params: Parámetros específicos del algoritmo
            
        Returns:
            Instancia configurada del algoritmo de clustering
        """
        if method == 'dbscan':
            return DBSCAN(n_jobs=self.n_jobs, **params)
        elif method == 'hdbscan':
            return hdbscan.HDBSCAN(**params, core_dist_n_jobs=self.n_jobs)
        elif method == 'optics':
            return OPTICS(n_jobs=self.n_jobs, **params)
        elif method == 'agglomerative':
            agg_params = params.copy()
            if 'distance_threshold' in agg_params and agg_params['distance_threshold'] is not None:
                agg_params['n_clusters'] = None
            return AgglomerativeClustering(**agg_params)
        else:
            raise ValueError(f"Método desconocido: {method}")
    
    def _optimize_parameters(self, method: str) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros para método de clustering específico usando Optuna.
        
        Args:
            method: Método de clustering a optimizar
            
        Returns:
            Mejores parámetros encontrados durante la optimización
        """
        def objective(trial):
            try:
                if method == 'dbscan':
                    param_ranges = self._get_dbscan_params_range()
                    params = {
                        'eps': trial.suggest_float('eps', *param_ranges['eps']),
                        'min_samples': trial.suggest_int('min_samples', *param_ranges['min_samples'])
                    }
                elif method == 'hdbscan':
                    param_ranges = self._get_hdbscan_params_range()
                    params = {
                        'min_cluster_size': trial.suggest_int('min_cluster_size', *param_ranges['min_cluster_size']),
                        'min_samples': trial.suggest_int('min_samples', *param_ranges['min_samples']),
                        'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', *param_ranges['cluster_selection_epsilon'])
                    }
                elif method == 'optics':
                    param_ranges = self._get_optics_params_range()
                    params = {
                        'min_samples': trial.suggest_int('min_samples', *param_ranges['min_samples']),
                        'xi': trial.suggest_float('xi', *param_ranges['xi']),
                        'min_cluster_size': trial.suggest_int('min_cluster_size', *param_ranges['min_cluster_size'])
                    }
                elif method == 'agglomerative':
                    param_ranges = self._get_agglomerative_params_range()
                    linkage = trial.suggest_categorical('linkage', param_ranges['linkage'])
                    distance_threshold = trial.suggest_float(
                        'distance_threshold', 
                        *param_ranges['distance_threshold']
                    )
                    params = {
                        'distance_threshold': distance_threshold,
                        'linkage': linkage,
                        'n_clusters': None 
                    }
                # Crear y ajustar clusterer
                clusterer = self._create_clusterer(method, params)
                cluster_labels = clusterer.fit_predict(self.embeddings)
                
                # Validaciones de clustering
                unique_labels = np.unique(cluster_labels)
                n_clusters = len(unique_labels[unique_labels != -1])  # Excluir ruido si existe
                n_samples = len(self.embeddings)
                
                # Filtros de validez más estrictos
                if n_clusters < 2 or n_clusters >= n_samples * 0.8:
                    return -1.0
                
                # Función objetivo multi-criterio optimizada
                try:
                    # Métricas de calidad interna
                    silhouette = silhouette_score(self.embeddings, cluster_labels)
                    calinski_harabasz = calinski_harabasz_score(self.embeddings, cluster_labels)
                    
                    # Métricas de correspondencia externa
                    ari = adjusted_rand_score(self.true_labels, cluster_labels)
                    nmi = normalized_mutual_info_score(self.true_labels, cluster_labels)
                    
                    # Normalización inteligente
                    silhouette_norm = (silhouette + 1) / 2  # [-1,1] -> [0,1]
                    calinski_norm = min(1.0, calinski_harabasz / 1000)  # Normalizar CH
                    ari_norm = max(0, ari)  # ARI puede ser negativo
                    nmi_norm = nmi  # NMI ya está en [0,1]
                    
                    # Función objetivo balanceada para embeddings de visión:
                    # 35% separabilidad + 25% compacidad + 25% ARI + 15% NMI
                    combined_score = (0.35 * silhouette_norm + 
                                    0.25 * calinski_norm +
                                    0.25 * ari_norm + 
                                    0.15 * nmi_norm)
                    
                except Exception as metric_error:
                    logging.warning(f"Error calculando métricas para {method}: {metric_error}")
                    return -1.0
                finally:
                    del clusterer
                    gc.collect()
                
                return combined_score
                
            except Exception as e:
                logging.warning(f"Trial failed for {method}: {e}")
                return -1.0
        
        # Configuración avanzada del optimizador
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=max(5, min(8, self.optimizer_trials // 3)),  # Más exploratorio
            multivariate=True,
            constant_liar=True,
            n_ei_candidates=24  # Mejora exploración
        )
        
        # Estudio con pruning adaptativo
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=3,
                max_resource=self.optimizer_trials // 2,
                reduction_factor=3
            )
        )
        
        # Optimización con manejo robusto de errores
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(
                    objective,
                    n_trials=self.optimizer_trials,
                    show_progress_bar=False,  # Menos verboso
                    n_jobs=1  # Evitar conflictos de paralelización
                )
        except KeyboardInterrupt:
            logging.info(f"Optimización de {method} interrumpida por usuario")
        except Exception as opt_error:
            logging.error(f"Error durante optimización de {method}: {opt_error}")
            return {}
            
        if study.best_value > 0:
            logging.info(f"Optimización {method} completada. Mejor valor: {study.best_value:.4f}")
        else:
            logging.warning(f"No se encontraron parámetros válidos para {method}")
        return study.best_params if study.best_value > 0 else {}
    
    def cluster(self, method: str, params: Dict[str, Any] = None) -> np.ndarray:
        """
        Aplica clustering con parámetros dados u optimizados.
        
        Args:
            method: Método de clustering a aplicar
            params: Parámetros específicos (None para auto-optimización)
            
        Returns:
            Array de etiquetas de cluster asignadas
        """
        if params is None:
            logging.info(f"Optimizando parámetros para {method.upper()}...")
            params = self._optimize_parameters(method)
            if not params:
                raise ValueError(f"No se pudieron encontrar parámetros válidos para {method}")
            self.best_params[method] = params
            logging.info(f"Mejores parámetros para {method}: {params}")
        
        clusterer = self._create_clusterer(method, params)
        cluster_labels = clusterer.fit_predict(self.embeddings)

        del clusterer
        gc.collect()
        return cluster_labels
    
    def cluster_all(self, methods: List[str] = None) -> Dict[str, np.ndarray]:
        """Apply all specified clustering methods with optimization."""
        if methods is None:
            methods = self.available_methods
        results = {}

        for method in methods:
            logging.info(f"Processing {method.upper()}...")
            try:
                cluster_labels = self.cluster(method)
                results[method] = cluster_labels
                self.cluster_labels[method] = cluster_labels
                
                metrics = self.evaluate_clustering(cluster_labels, method)
                logging.info(f"{method.upper()} - Clusters: {len(np.unique(cluster_labels))}, "
                    f"Silhouette: {metrics['silhouette_score']:.4f}, "
                    f"ARI: {metrics['adjusted_rand_score']:.4f}")
                gc.collect()
    
            except Exception as e:
                logging.info(f"Failed to process {method}: {e}")
                continue
        self.results = results
        return results
    
    def evaluate_clustering(self, cluster_labels: np.ndarray, method_name: str = "") -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics."""
        metrics = {}
        cluster_labels = cluster_labels.astype(np.int32)
        valid_mask = cluster_labels != -1
        n_valid = np.sum(valid_mask)

        if n_valid < 2: 
            return {
                'silhouette_score': -1,
                'adjusted_rand_score': -1,
                'normalized_mutual_info': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': float('inf'),
                'n_clusters': 1,
                'n_noise_points': self.n_true_clusters - n_valid,
                'purity': 0.0
            }

        valid_embeddings = self.embeddings[valid_mask].astype(np.float64)  
        valid_cluster_labels = cluster_labels[valid_mask]
        valid_true_labels = self.true_labels[valid_mask]

        n_clusters = len(np.unique(valid_cluster_labels))
        if n_clusters > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(
                    valid_embeddings, 
                    valid_cluster_labels
                )
            except Exception as e:
                logging.warning(f"Silhouette score calculation failed for {method_name}: {e}")
                metrics['silhouette_score'] = -1
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                    valid_embeddings, 
                    valid_cluster_labels
                )
            except Exception as e:
                logging.warning(f"Calinski Harabasz score calculation failed for {method_name}: {e}")
                metrics['calinski_harabasz_score'] = -1
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(
                    valid_embeddings, 
                    valid_cluster_labels
                )
            except Exception as e:
                logging.warning(f"Davies Bouldin score calculation failed for {method_name}: {e}")
                metrics['davies_bouldin_score'] = float('inf')
        else:
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = -1
            metrics['davies_bouldin_score'] = float('inf')
        try:
            metrics['adjusted_rand_score'] = adjusted_rand_score(valid_true_labels, valid_cluster_labels)
        except Exception as e:
            logging.warning(f"Adjusted rand score calculation failed for {method_name}: {e}")
            metrics['adjusted_rand_score'] = -1
        try:
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(valid_true_labels, valid_cluster_labels)
        except Exception as e:
            logging.warning(f"Normalized mutual info calculation failed for {method_name}: {e}")
            metrics['normalized_mutual_info'] = -1   
        try:
            metrics['purity'] = fast_purity_calculation(cluster_labels, self.true_labels)
        except Exception as e:
            logging.warning(f"Purity calculation failed for {method_name}: {e}")
            metrics['purity'] = 0.0

        metrics['n_clusters'] = n_clusters
        metrics['n_noise_points'] = len(self.true_labels) - n_valid
        return metrics
    
    def compare_methods(self, methods: List[str] = None) -> pd.DataFrame:
        """Compare different clustering methods using multiple metrics."""
        if not self.results:
            self.cluster_all(methods)
        
        comparison_data = []
        logging.info("\nDetailed Comparison Results:")
        logging.info("=" * 80)
        
        for method, cluster_labels in self.results.items():
            metrics = self.evaluate_clustering(cluster_labels, method)
            
            row_data = {'method': method}
            row_data.update(metrics)
            comparison_data.append(row_data)
            
            logging.info(f"\n{method.upper()}:")
            logging.info(f"  Clusters: {metrics['n_clusters']}")
            if method == 'dbscan':
                logging.info(f"  Noise points: {metrics['n_noise_points']}")
            logging.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
            logging.info(f"  Adjusted Rand Index: {metrics['adjusted_rand_score']:.4f}")
            logging.info(f"  Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
            logging.info(f"  Purity: {metrics['purity']:.4f}")
        
        comparison_df = pd.DataFrame(comparison_data)
        logging.info(f"\n{'BEST METHODS':=^80}")
        metrics_to_rank = ['silhouette_score', 'adjusted_rand_score', 'normalized_mutual_info', 'calinski_harabasz_score', 'purity']
        for metric in metrics_to_rank:
            try:
                best_idx = comparison_df[metric].idxmax()
                best_method = comparison_df.loc[best_idx, 'method']
                best_value = comparison_df.loc[best_idx, metric]
                logging.info(f"{metric:25}: {best_method.upper():15} ({best_value:.4f})")
            except Exception:
                continue   

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
            else:  
                if score > best_score:
                    best_score = score
                    best_method = method

        return best_method, self.results[best_method]
    
    def get_cluster_summary(self, method: str) -> pd.DataFrame:
        """Get summary statistics for each cluster efficiently."""
        if method not in self.results:
            raise ValueError(f"Method {method} not found. Available: {list(self.results.keys())}")
        
        cluster_labels = self.results[method]
        summary_data = []
        
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_true_labels = self.true_labels[mask]
            
            if len(cluster_true_labels) == 0:
                continue
            
            true_label_counts = pd.Series(cluster_true_labels).value_counts()
            most_common_true_label = true_label_counts.index[0]
            
            summary_data.append(
                {
                    'cluster_id': cluster_id,
                    'size': np.sum(mask),
                    'most_common_true_label': most_common_true_label,
                    'purity': true_label_counts.iloc[0] / np.sum(mask),
                    'n_unique_true_labels': len(true_label_counts)
                }
            )
        
        return pd.DataFrame(summary_data).sort_values('cluster_id')