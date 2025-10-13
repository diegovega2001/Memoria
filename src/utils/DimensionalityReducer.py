"""
Reductor de dimensionalidad optimizado para análisis de clustering de embeddings.

Este módulo proporciona métodos de reducción de dimensionalidad (PCA, t-SNE, UMAP)
optimizados específicamente para preservar la estructura necesaria para clustering.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import torch
import umap
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class DimensionalityReducer:
    """
    Reductor de dimensionalidad optimizado para análisis de clustering de embeddings.
    
    Proporciona métodos de reducción (PCA, t-SNE, UMAP) con optimización automática
    de hiperparámetros enfocada en preservar la estructura de clustering.
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        seed: int = 3,
        optimizer_trials: int = 50,
        available_methods: List[str] = None,
        use_incremental: bool = True,
        n_jobs: int = -1,
        batch_size: int = 1000
    ):
        """
        Inicializa el reductor de dimensionalidad.
        
        Args:
            embeddings: Tensor de embeddings a reducir
            labels: Etiquetas verdaderas para evaluación
            seed: Semilla para reproducibilidad
            optimizer_trials: Número de trials para optimización
            available_methods: Lista de métodos disponibles
            use_incremental: Si usar PCA incremental para datasets grandes
            n_jobs: Número de cores para paralelización
            batch_size: Tamaño de batch para procesamiento incremental
        """
        if available_methods is None:
            available_methods = ['pca', 'umap', 'tsne']
            
        # Conversión y preprocesamiento de datos
        if hasattr(embeddings, 'cpu'):  # Tensor de PyTorch
            self.embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            self.embeddings = embeddings.astype(np.float32)
            
        if hasattr(labels, 'cpu'):  # Tensor de PyTorch
            self.labels = labels.cpu().numpy().astype(np.int32)
        else:
            self.labels = labels.astype(np.int32)
        
        # Normalización estándar para mejores resultados
        self.scaler = StandardScaler()
        self.embeddings = self.scaler.fit_transform(self.embeddings)
        
        # Configuración
        self.seed = seed
        self.optimizer_trials = optimizer_trials
        self.available_methods = available_methods
        self.use_incremental = use_incremental
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count() - 1
        
        # Resultados y parámetros óptimos
        self.results: Dict[str, np.ndarray] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        
        # Liberación de memoria GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"Inicializado con {self.embeddings.shape[0]} muestras, {self.embeddings.shape[1]} características")
        logging.info(f"Usando {self.n_jobs} núcleos de CPU")        
    
    def _get_pca_params_range(self) -> Dict[str, Tuple[int, int]]:
        n_samples, n_features = self.embeddings.shape
        max_possible = min(n_features - 1, n_samples - 1)
        min_components = max(16, min(32, max_possible // 4))
        max_components = min(max_possible, max(512, int(n_features * 0.9)))
        return {'n_components': (min_components, max_components)}
    
    def _get_tsne_params_range(self) -> Dict[str, Tuple]:
        n_samples = len(self.embeddings)
        max_perplexity = min(100, (n_samples - 1) // 3)
        min_perplexity = max(5, min(10, n_samples // 150))
        return {
            'n_components': (2, 3),
            'perplexity': (min_perplexity, max_perplexity),
            'learning_rate': (10.0, 1000.0),
            'max_iter': (500, 3000),
            'early_exaggeration': (4.0, 36.0)
        }
    
    def _get_umap_params_range(self) -> Dict[str, Tuple]:
        n_samples = len(self.embeddings)
        max_neighbors = min(300, n_samples - 1)
        min_neighbors = max(5, min(10, n_samples // 100))
        return {
            'n_components': (2, 50),
            'n_neighbors': (min_neighbors, max_neighbors),
            'min_dist': (0.0, 0.8),
            'metric': ['euclidean', 'cosine', 'manhattan'],
            'n_epochs': (200, 1500),
            'learning_rate': (0.1, 3.0),
            'negative_sample_rate': (2, 15)
        }
    
    def _create_reducer(self, method: str, params: Dict[str, Any]):
        """
        Crea una instancia del reductor con los parámetros dados.
        
        Args:
            method: Método de reducción ('pca', 'tsne', 'umap')
            params: Diccionario con parámetros del método
            
        Returns:
            Instancia del reductor configurada
        """
        if method == 'pca':
            # Usar PCA incremental para datasets grandes
            if self.use_incremental and self.embeddings.shape[0] > 10000:
                return IncrementalPCA(
                    n_components=params['n_components'],
                    batch_size=min(self.batch_size, self.embeddings.shape[0] // 10)
                )
            else:
                return PCA(random_state=self.seed, **params)
        elif method == 'tsne':
            return TSNE(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                **params
            )
        elif method == 'umap':
            return umap.UMAP(
                random_state=self.seed,
                n_jobs=self.n_jobs,
                low_memory=True,
                **params
            )
        else:
            raise ValueError(f"Método desconocido: {method}")
    
    def _optimize_parameters(self, method: str) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros para un método específico usando Optuna.
        
        Args:
            method: Método de reducción a optimizar
            
        Returns:
            Diccionario con los mejores parámetros encontrados
        """
        def objective(trial):
            """Función objetivo para optimización con Optuna."""
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
                        'learning_rate': trial.suggest_float('learning_rate', *param_ranges['learning_rate'], log=True),
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
                        'metric': trial.suggest_categorical('metric', param_ranges['metric']),
                        'n_epochs': trial.suggest_int('n_epochs', *param_ranges['n_epochs']),
                        'negative_sample_rate': trial.suggest_int('negative_sample_rate', *param_ranges['negative_sample_rate'])
                    }
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                reducer = self._create_reducer(method, params)
                reduced_embeddings = reducer.fit_transform(self.embeddings)
                
                if reduced_embeddings.shape[0] < 2:
                    return -1.0
                
                silhouette_val = silhouette_score(reduced_embeddings, self.labels)
                
                if method in ['umap', 'tsne']:
                    trust_val = trustworthiness(self.embeddings, reduced_embeddings, n_neighbors=min(5, len(self.embeddings) - 1))
                    silhouette_norm = (silhouette_val + 1) / 2
                    combined_score = 0.7 * silhouette_norm + 0.3 * trust_val
                else:
                    combined_score = (silhouette_val + 1) / 2
                
                del reducer, reduced_embeddings
                gc.collect()
                
                return combined_score
                
            except Exception as e:
                logging.debug(f"Trial failed for {method}: {e}")
                return -1.0
        
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=max(10, self.optimizer_trials // 5),
            multivariate=True,
            constant_liar=True
        )
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=8,
                n_warmup_steps=12
            )
        )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                study.optimize(
                    objective,
                    n_trials=self.optimizer_trials,
                    show_progress_bar=False,
                    catch=(Exception,)
                )
        except KeyboardInterrupt:
            logging.info(f"Optimization interrupted for {method}")
        
        if study.best_value > 0:
            logging.info(f"{method.upper()} optimization: Best score {study.best_value:.4f}")
            logging.info(f"Best parameters: {study.best_params}")
        else:
            logging.warning(f"No valid parameters found for {method}")
        
        return study.best_params if study.best_value > 0 else {}
    
    def reduce(self, method: str, params: Dict[str, Any] = None) -> np.ndarray:
        """
        Aplica reducción de dimensionalidad con parámetros dados u optimizados.
        
        Args:
            method: Método de reducción a aplicar
            params: Parámetros específicos (si None, se optimizan automáticamente)
            
        Returns:
            Array numpy con embeddings reducidos
        """
        if params is None:
            logging.info(f"Optimizando parámetros para {method.upper()}...")
            params = self._optimize_parameters(method)
            self.best_params[method] = params
        
        # Crear y aplicar reductor
        reducer = self._create_reducer(method, params)
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        # Limpieza de memoria
        del reducer
        gc.collect()
        
        return reduced_embeddings
    
    def reduce_all(self, methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Aplica todos los métodos de reducción especificados con optimización.
        
        Args:
            methods: Lista de métodos a aplicar (si None, usa available_methods)
            
        Returns:
            Diccionario con resultados de cada método
        """
        if methods is None:
            methods = self.available_methods
            
        results = {}
        
        for method in methods:
            logging.info(f"Procesando {method.upper()}...")
            try:
                reduced_embeddings = self.reduce(method)
                results[method] = reduced_embeddings
                
                # Evaluar calidad
                score = silhouette_score(reduced_embeddings, self.labels)
                logging.info(f"{method.upper()} silhouette score: {score:.4f}")
                
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error procesando {method}: {e}")
                continue
        
        self.results = results
        return results
    
    def compare_methods(self, methods: List[str] = None) -> Dict[str, float]:
        """
        Compara diferentes métodos de reducción usando silhouette score.
        
        Args:
            methods: Métodos a comparar (si None, usa resultados existentes)
            
        Returns:
            Diccionario con scores de cada método
        """
        if not self.results:
            self.reduce_all(methods)
        
        scores = {}
        logging.info("\nResultados de Comparación:")
        logging.info("-" * 40)
        
        for method, embeddings in self.results.items():
            score = silhouette_score(embeddings, self.labels)
            scores[method] = score
            logging.info(f"{method.upper():>8}: {score:.4f}")
            
        return scores
    
    def get_best_result(self) -> Tuple[str, np.ndarray]:
        """
        Obtiene el mejor resultado de reducción basado en silhouette score.
        
        Returns:
            Tupla con (mejor_método, embeddings_reducidos)
        """
        if not self.results:
            raise ValueError("No hay resultados disponibles. Ejecute reduce_all() primero.")
        
        scores = self.compare_methods()
        best_method = max(scores.keys(), key=lambda k: scores[k])
        
        return best_method, self.results[best_method]