"""
Visualizador de clusters optimizado para análisis de embeddings.

Este módulo proporciona herramientas de visualización avanzada para resultados
de clustering, incluyendo reducción de dimensionalidad y análisis de pureza.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class ClusterVisualizer:
    """
    Visualizador avanzado de resultados de clustering.
    
    Proporciona métodos optimizados para visualización de clusters mediante
    reducción de dimensionalidad, análisis de pureza y generación de gráficos
    de alta calidad para análisis de embeddings de visión computacional.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray,
        test_dataset,
        label_encoder,
        seed: int
    ):
        """
        Inicializa el visualizador de clusters.
        
        Args:
            embeddings: Array de embeddings a visualizar
            cluster_labels: Etiquetas de cluster asignadas
            true_labels: Etiquetas verdaderas para comparación
            test_dataset: Dataset de test para acceso a imágenes
            label_encoder: Codificador de etiquetas para nombres de clases
            seed: Semilla para reproducibilidad
        """
        # Conversión y preprocesamiento de datos
        if hasattr(embeddings, 'cpu'):  # Tensor de PyTorch
            self.embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            self.embeddings = embeddings.astype(np.float32)
            
        if hasattr(cluster_labels, 'cpu'):  # Tensor de PyTorch
            self.cluster_labels = cluster_labels.cpu().numpy().astype(np.int32)
        else:
            self.cluster_labels = cluster_labels.astype(np.int32)
            
        if hasattr(true_labels, 'cpu'):  # Tensor de PyTorch
            self.true_labels = true_labels.cpu().numpy().astype(np.int32)
        else:
            self.true_labels = true_labels.astype(np.int32)
        
        # Configuración
        self.test_dataset = test_dataset
        self.label_encoder = label_encoder
        self.seed = seed
        
        # Análisis automático de pureza
        self.cluster_analysis = self._analyze_cluster_purity()
        self.model_names = self.label_encoder.classes_
        
        logging.info(f"Inicializado visualizador: {len(self.embeddings)} embeddings, "
                    f"{len(np.unique(self.cluster_labels))} clusters")
        
    def _analyze_cluster_purity(self) -> Dict[int, Dict[str, Union[int, float, str]]]:
        """
        Analiza la pureza de cada cluster identificado.
        
        Calcula estadísticas detalladas para cada cluster incluyendo tamaño,
        diversidad de modelos, pureza y distribución de clases verdaderas.
        
        Returns:
            Diccionario con análisis completo por cluster
        """
        unique_clusters = np.unique(self.cluster_labels)
        cluster_info = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Saltar puntos de ruido
                continue
                
            mask = self.cluster_labels == cluster_id
            cluster_true_labels = self.true_labels[mask]
            
            if len(cluster_true_labels) == 0:
                continue
            
            unique_models = np.unique(cluster_true_labels)
            model_counts = pd.Series(cluster_true_labels).value_counts()
            
            # Estadísticas de pureza mejoradas
            cluster_size = np.sum(mask)
            most_common_count = model_counts.iloc[0]
            purity = most_common_count / cluster_size
            
            cluster_info[cluster_id] = {
                'size': cluster_size,
                'n_unique_models': len(unique_models),
                'most_common_model': model_counts.index[0],
                'most_common_count': most_common_count,
                'purity': purity,
                'diversity_index': 1.0 - purity,  # Índice de diversidad (1 - pureza)
                'entropy': -np.sum((model_counts / cluster_size) * np.log2(model_counts / cluster_size + 1e-8)),
                'model_distribution': model_counts.to_dict(),
                'is_pure': len(unique_models) == 1,
                'is_dominant': purity >= 0.8  # Cluster con > 80% de una clase
            }
        
        return cluster_info
    
    def plot_embeddings_overview(
        self, 
        reduction_method: str,
        color_by: str, 
        figsize: Tuple[int, int],
        alpha: float,
        s: int
    ):
        """
        Genera visualización general de embeddings en espacio 2D.
        
        Aplica reducción de dimensionalidad y crea gráficos comparativos
        mostrando tanto clusters identificados como etiquetas verdaderas.
        
        Args:
            reduction_method: Método de reducción ('tsne', 'pca', 'umap')
            color_by: Criterio de coloreo ('cluster', 'true', 'both')
            figsize: Tamaño de la figura (ancho, alto)
            alpha: Transparencia de los puntos
            s: Tamaño de los puntos
        """
        # Configuración adaptativa del reductor
        n_samples = len(self.embeddings)
        
        if reduction_method == 'tsne':
            if n_samples < 4:
                logging.warning("Dataset muy pequeño, usando PCA en su lugar")
                reducer = PCA(n_components=2, random_state=self.seed)
            else:
                perplexity = max(5, min(50, n_samples // 8))  # Perplexity adaptativa
                reducer = TSNE(
                    n_components=2, 
                    random_state=self.seed, 
                    perplexity=perplexity,
                    n_iter=1000,  # Más iteraciones para convergencia
                    learning_rate='auto'
                )
        elif reduction_method == 'pca':
            reducer = PCA(n_components=2, random_state=self.seed)
        elif reduction_method == 'umap':
            n_neighbors = max(5, min(15, n_samples // 20))  # Neighbors adaptativos
            reducer = umap.UMAP(
                n_components=2, 
                random_state=self.seed,
                n_neighbors=n_neighbors,
                min_dist=0.1
            )
        else:
            raise ValueError(f"Método de reducción no soportado: {reduction_method}")
        
        # Aplicar reducción de dimensionalidad
        logging.info(f"Aplicando {reduction_method.upper()} a {n_samples} embeddings...")
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        # Crear visualización comparativa
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Gráfico por clusters
        unique_clusters = np.unique(self.cluster_labels)
        n_clusters = len(unique_clusters)
        
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.cluster_labels, 
            cmap='tab10' if n_clusters <= 10 else 'tab20',
            alpha=alpha, s=s, edgecolors='black', linewidth=0.5
        )
        axes[0].set_title('Coloreado por Etiquetas de Cluster', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'{reduction_method.upper()} Componente 1', fontsize=12)
        axes[0].set_ylabel(f'{reduction_method.upper()} Componente 2', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico por etiquetas verdaderas
        unique_true = np.unique(self.true_labels)
        n_true = len(unique_true)
        
        scatter2 = axes[1].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.true_labels, 
            cmap='tab20' if n_true <= 20 else 'viridis',
            alpha=alpha, s=s, edgecolors='black', linewidth=0.5
        )
        axes[1].set_title('Coloreado por Etiquetas Verdaderas', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'{reduction_method.upper()} Componente 1', fontsize=12)
        axes[1].set_ylabel(f'{reduction_method.upper()} Componente 2', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Añadir información estadística
        fig.suptitle(f'Visualización de Embeddings - {n_samples} muestras, '
                    f'{n_clusters} clusters, {n_true} clases verdaderas', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.show()
        
        # Limpieza de memoria
        del reducer
        gc.collect()
        
        return embeddings_2d
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """
        Genera resumen del análisis de pureza de clusters.
        
        Returns:
            DataFrame con estadísticas completas por cluster
        """
        summary_data = []
        
        for cluster_id, info in self.cluster_analysis.items():
            model_name = self.model_names[info['most_common_model']]
            summary_data.append({
                'cluster_id': cluster_id,
                'size': info['size'],
                'n_unique_models': info['n_unique_models'],
                'most_common_model': model_name,
                'purity': info['purity'],
                'diversity_index': info['diversity_index'],
                'entropy': info['entropy'],
                'is_pure': info['is_pure'],
                'is_dominant': info['is_dominant']
            })
        
        df = pd.DataFrame(summary_data).sort_values('purity', ascending=False)
        logging.info(f"Generado resumen: {len(df)} clusters analizados")
        return df
    
    def visualize_good_clusters(
        self, 
        n_clusters: int, 
        images_per_cluster: int,
        figsize_per_cluster: Tuple[int, int]
    ):
        """
        Visualiza los clusters más puros con imágenes representativas.
        
        Selecciona automáticamente los clusters con mayor pureza y muestra
        imágenes representativas de cada uno para análisis visual.
        
        Args:
            n_clusters: Número de clusters a visualizar
            images_per_cluster: Imágenes por cluster a mostrar
            figsize_per_cluster: Tamaño de figura por cluster
        """
        # Seleccionar clusters de alta calidad (pureza > 0.7 o dominantes)
        high_quality_clusters = [
            (cluster_id, info) for cluster_id, info in self.cluster_analysis.items()
            if info['purity'] >= 0.7 or info['is_dominant']
        ]
        
        # Ordenar por pureza descendente
        high_quality_clusters.sort(key=lambda x: x[1]['purity'], reverse=True)
        
        if len(high_quality_clusters) == 0:
            logging.warning("No se encontraron clusters de alta calidad!")
            return
        
        # Limitar número de clusters a visualizar
        n_clusters = min(n_clusters, len(high_quality_clusters))
        logging.info(f"Visualizando {n_clusters} clusters de mayor pureza")

        # Configurar grid de visualización
        fig, axes = plt.subplots(n_clusters, 1, 
                                figsize=(figsize_per_cluster[0], figsize_per_cluster[1] * n_clusters))
        if n_clusters == 1:
            axes = [axes]
        
        # Visualizar cada cluster seleccionado
        for i, (cluster_id, cluster_info) in enumerate(high_quality_clusters[:n_clusters]):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Selección diversificada de imágenes
            n_to_show = min(images_per_cluster, len(cluster_indices))
            np.random.seed(self.seed + i)  # Semilla consistente por cluster
            selected_indices = np.random.choice(cluster_indices, n_to_show, replace=False)
            
            model_name = self.model_names[cluster_info['most_common_model']]
            
            # Título descriptivo con estadísticas
            title = (f"Cluster {cluster_id} - Modelo: {model_name} | "
                    f"Tamaño: {cluster_info['size']} | "
                    f"Pureza: {cluster_info['purity']:.3f}")
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].axis('off')
            
            # Mostrar imágenes representativas del cluster
            for j, idx in enumerate(selected_indices):
                try:
                    sample = self.test_dataset[idx]
                    images = sample['images']
                    
                    # Manejar diferentes formatos de imagen
                    if isinstance(images, list):
                        img = images[0]  # Tomar primera imagen si es lista
                    else:
                        img = images
                    
                    # Conversión y normalización robusta
                    if hasattr(img, 'numpy'):  # Tensor de PyTorch
                        img_np = img.permute(1, 2, 0).numpy()
                        # Desnormalización ImageNet
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = img_np * std + mean
                        img_np = np.clip(img_np, 0, 1)
                    else:  # PIL Image o numpy array
                        img_np = np.array(img)
                        if img_np.max() > 1.0:  # Si está en rango [0,255]
                            img_np = img_np / 255.0
                    
                    # Crear subgráfico para imagen
                    ax_img = plt.subplot2grid(
                        (n_clusters, images_per_cluster), (i, j),
                        fig=fig
                    )
                    ax_img.imshow(img_np)
                    ax_img.axis('off')
                    ax_img.set_title(f'Muestra {j+1}', fontsize=10)
                    
                except Exception as img_error:
                    logging.warning(f"Error procesando imagen {idx}: {img_error}")
                    continue
        
        plt.suptitle(f'Clusters de Alta Pureza - Top {n_clusters}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
        logging.info(f"Visualización completada para {n_clusters} clusters")
    
    def visualize_mixed_clusters(
        self, 
        n_clusters: int,
        max_models_per_cluster: int,
        figsize_per_cluster: Tuple[int, int]
    ):
        """
        Visualiza clusters mixtos mostrando diversidad de modelos.
        
        Selecciona clusters con mayor diversidad y muestra una imagen
        representativa de cada modelo presente en el cluster.
        
        Args:
            n_clusters: Número de clusters mixtos a visualizar
            max_models_per_cluster: Máximo de modelos diferentes a mostrar por cluster
            figsize_per_cluster: Tamaño de figura por cluster
        """
        # Seleccionar clusters mixtos con mayor diversidad
        mixed_clusters = [
            (cluster_id, info) for cluster_id, info in self.cluster_analysis.items() 
            if not info['is_pure'] and info['n_unique_models'] > 1
        ]
        
        # Ordenar por diversidad (número de modelos únicos)
        mixed_clusters.sort(key=lambda x: x[1]['n_unique_models'], reverse=True)
        
        if len(mixed_clusters) == 0:
            logging.warning("No se encontraron clusters mixtos!")
            return
        
        # Limitar número de clusters a visualizar
        n_clusters = min(n_clusters, len(mixed_clusters))
        logging.info(f"Visualizando {n_clusters} clusters mixtos más diversos")

        # Configurar grid de visualización
        fig, axes = plt.subplots(n_clusters, 1, 
                                figsize=(figsize_per_cluster[0], figsize_per_cluster[1] * n_clusters))
        if n_clusters == 1:
            axes = [axes]
        
        # Procesar cada cluster mixto seleccionado
        for i, (cluster_id, cluster_info) in enumerate(mixed_clusters[:n_clusters]):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_true_labels = self.true_labels[cluster_mask]
            
            # Obtener modelos únicos en el cluster
            unique_models = np.unique(cluster_true_labels)
            models_to_show = unique_models[:max_models_per_cluster]
            
            # Título descriptivo con estadísticas
            title = (f"Cluster Mixto {cluster_id} - {len(unique_models)} modelos | "
                    f"Tamaño: {cluster_info['size']} | "
                    f"Pureza: {cluster_info['purity']:.3f} | "
                    f"Entropía: {cluster_info['entropy']:.3f}")
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis('off')
            
            # Mostrar una imagen representativa por cada modelo
            for j, model_label in enumerate(models_to_show):
                # Encontrar índices de este modelo en el cluster
                model_mask = cluster_true_labels == model_label
                local_indices = np.where(model_mask)[0]
                if len(local_indices) == 0:
                    continue
                
                try:
                    # Seleccionar imagen representativa del modelo
                    np.random.seed(self.seed + int(model_label))
                    selected_local_idx = np.random.choice(local_indices)
                    global_idx = cluster_indices[selected_local_idx]
                    
                    sample = self.test_dataset[global_idx]
                    images = sample['images']
                    
                    # Manejar diferentes formatos de imagen
                    if isinstance(images, list):
                        img = images[0]  
                    else:
                        img = images
                    
                    # Conversión y normalización robusta
                    if hasattr(img, 'numpy'):  # Tensor de PyTorch
                        img_np = img.permute(1, 2, 0).numpy()
                        # Desnormalización ImageNet
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = img_np * std + mean
                        img_np = np.clip(img_np, 0, 1)
                    else:  # PIL Image o numpy array
                        img_np = np.array(img)
                        if img_np.max() > 1.0:
                            img_np = img_np / 255.0
                    
                    # Crear subgráfico para imagen
                    ax_img = plt.subplot2grid(
                        (n_clusters, max_models_per_cluster), (i, j),
                        fig=fig
                    )
                    ax_img.imshow(img_np)
                    ax_img.axis('off')
                    
                    # Etiqueta con nombre del modelo y cantidad de imágenes
                    model_name = self.model_names[model_label]
                    count = cluster_info['model_distribution'].get(model_label, 0)
                    percentage = (count / cluster_info['size']) * 100
                    ax_img.set_title(f'{model_name}\n({count} imgs, {percentage:.1f}%)', 
                                    fontsize=9, fontweight='bold')
                    
                except Exception as img_error:
                    logging.warning(f"Error procesando imagen para modelo {model_label}: {img_error}")
                    continue
        
        plt.suptitle(f'Clusters Mixtos - Diversidad de Modelos', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
        logging.info(f"Visualización completada para {n_clusters} clusters mixtos")
    
    def print_cluster_statistics(self):
        """
        Imprime estadísticas generales sobre la calidad del clustering.
        
        Genera un resumen completo de métricas de pureza, diversidad
        y distribución de clusters para análisis cuantitativo.
        """
        # Estadísticas básicas
        total_clusters = len(self.cluster_analysis)
        if total_clusters == 0:
            logging.warning("No hay clusters para analizar")
            return
            
        pure_clusters = sum(1 for info in self.cluster_analysis.values() if info['is_pure'])
        dominant_clusters = sum(1 for info in self.cluster_analysis.values() if info['is_dominant'])
        mixed_clusters = total_clusters - pure_clusters
        
        # Estadísticas de pureza
        purities = [info['purity'] for info in self.cluster_analysis.values()]
        entropies = [info['entropy'] for info in self.cluster_analysis.values()]
        sizes = [info['size'] for info in self.cluster_analysis.values()]
        
        # Reporte principal
        logging.info("\n" + "=" * 60)
        logging.info("               ESTADÍSTICAS DE CLUSTERING")
        logging.info("=" * 60)
        logging.info(f"Total de clusters:        {total_clusters}")
        logging.info(f"Clusters puros:          {pure_clusters} ({pure_clusters/total_clusters*100:.1f}%)")
        logging.info(f"Clusters dominantes:     {dominant_clusters} ({dominant_clusters/total_clusters*100:.1f}%)")
        logging.info(f"Clusters mixtos:         {mixed_clusters} ({mixed_clusters/total_clusters*100:.1f}%)")
        
        # Métricas de calidad
        logging.info(f"\nMÉTRICAS DE CALIDAD:")
        logging.info(f"Pureza promedio:         {np.mean(purities):.3f} ± {np.std(purities):.3f}")
        logging.info(f"Entropía promedio:       {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        logging.info(f"Tamaño promedio:         {np.mean(sizes):.1f} ± {np.std(sizes):.1f}")
        
        # Análisis de clusters mixtos
        if mixed_clusters > 0:
            mixed_models = [
                info['n_unique_models'] for info in self.cluster_analysis.values() 
                if not info['is_pure']
            ]
            logging.info(f"\nANÁLISIS DE CLUSTERS MIXTOS:")
            logging.info(f"Modelos promedio por cluster mixto: {np.mean(mixed_models):.1f}")
            logging.info(f"Máximo modelos en cluster mixto:    {max(mixed_models)}")
            
            # Top clusters más diversos
            mixed_info = [
                (cluster_id, info) for cluster_id, info in self.cluster_analysis.items()
                if not info['is_pure']
            ]
            mixed_info.sort(key=lambda x: x[1]['n_unique_models'], reverse=True)
            
            logging.info(f"\nTOP 3 CLUSTERS MÁS DIVERSOS:")
            for i, (cluster_id, info) in enumerate(mixed_info[:3]):
                logging.info(f"  {i+1}. Cluster {cluster_id}: {info['n_unique_models']} modelos, "
                           f"Pureza: {info['purity']:.3f}, Tamaño: {info['size']}")
        
        # Distribución de tamaños
        logging.info(f"\nDISTRIBUCIÓN DE TAMAÑOS:")
        logging.info(f"Cluster más pequeño:     {min(sizes)}")
        logging.info(f"Cluster más grande:      {max(sizes)}")
        logging.info(f"Mediana de tamaño:       {np.median(sizes):.1f}")
        
        logging.info("=" * 60)