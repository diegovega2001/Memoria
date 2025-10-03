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
        """
        Define rangos de parámetros para optimización de PCA.
        
        Optimizado para preservar estructura de clustering manteniendo
        suficientes componentes para separabilidad.
        """
        n_samples, n_features = self.embeddings.shape
        max_possible = min(n_features - 1, n_samples - 1)
        
        # Rangos conservadores: preservar más varianza para mejor clustering
        min_components = max(20, min(50, max_possible // 3))
        max_components = min(max_possible, max(300, int(n_features * 0.85)))
        
        return {'n_components': (min_components, max_components)}
    
    def _get_tsne_params_range(self) -> Dict[str, Tuple]:
        """
        Define rangos de parámetros para optimización de t-SNE.
        
        Enfocado en calidad de clustering con balance entre velocidad y calidad.
        """
        n_samples = self.embeddings.shape[0]
        
        # Perplexity adaptativo al tamaño del dataset
        perplexity_min = max(15, min(30, n_samples // 80))
        perplexity_max = min(200, n_samples // 3)
        
        return {
            'n_components': (2, 30),  # Hasta 30D para análisis más rico
            'perplexity': (perplexity_min, perplexity_max),
            'learning_rate': (100, 800),  # Rango más amplio para convergencia
            'max_iter': (750, 1500),  # Balance velocidad-calidad
            'early_exaggeration': (8.0, 20.0)
        }
    
    def _get_umap_params_range(self) -> Dict[str, Tuple]:
        """
        Define rangos de parámetros para optimización de UMAP.
        
        Optimizado para preservar estructura global necesaria para clustering
        manteniendo eficiencia computacional.
        """
        n_samples = self.embeddings.shape[0]
        
        # Vecinos adaptativos: más vecinos preservan mejor la estructura global
        neighbors_min = max(10, min(25, n_samples // 80))
        neighbors_max = min(150, n_samples // 8)
        
        return {
            'n_components': (2, min(64, self.embeddings.shape[1] // 2)),
            'n_neighbors': (neighbors_min, neighbors_max),
            'min_dist': (0.0, 0.4),  # Rango ampliado para explorar más separación
            'learning_rate': (0.3, 3.0),  # Rango más amplio para convergencia
            'metric': ['euclidean', 'cosine', 'manhattan']  # Métricas más diversas
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
                # Definir parámetros según el método
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
                else:
                    raise ValueError(f"Método no soportado: {method}")
                
                # Crear reductor y aplicar transformación
                reducer = self._create_reducer(method, params)
                reduced_embeddings = reducer.fit_transform(self.embeddings)
                
                # Función objetivo enfocada en clustering de embeddings
                silhouette_val = silhouette_score(reduced_embeddings, self.labels)
                
                if method in ['umap', 'tsne']:  # Métodos no lineales
                    # Para métodos no lineales, balancear separabilidad y preservación
                    trust_val = trustworthiness(self.embeddings, reduced_embeddings, n_neighbors=5)
                    silhouette_norm = (silhouette_val + 1) / 2
                    # 75% separabilidad + 25% preservación de estructura
                    combined_score = 0.75 * silhouette_norm + 0.25 * trust_val
                else:  # PCA
                    # Para PCA, solo usar silhouette (ya preserva estructura linealmente)
                    combined_score = (silhouette_val + 1) / 2
                
                # Limpieza de memoria
                del reducer, reduced_embeddings
                gc.collect()
                
                return combined_score
                
            except Exception as e:
                logging.warning(f"Trial falló para {method}: {e}")
                return -1.0
        
        # Configurar sampler optimizado para eficiencia
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=max(3, min(8, self.optimizer_trials // 4)),
            multivariate=True,
            constant_liar=True  # Mejora paralelización
        )
        
        # Crear estudio con pruner agresivo
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=3,  # Mínimo de evaluaciones antes de podar
                max_resource=self.optimizer_trials,
                reduction_factor=3
            )
        )
        
        try:
            study.optimize(
                objective,
                n_trials=self.optimizer_trials,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            logging.info("Optimización interrumpida por el usuario")
            
        logging.info(f"Optimización completada para {method}. Mejor valor: {study.best_value:.4f}")
        logging.info(f"Mejores parámetros para {method}: {study.best_params}")
        
        return study.best_params
    
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