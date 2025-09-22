import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Dict, List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import logging


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class ClusterVisualizer:
    def __init__(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: np.ndarray,
        test_dataset, 
        label_encoder,  
        seed: int = 3
    ):
        self.embeddings = embeddings
        self.cluster_labels = cluster_labels
        self.true_labels = true_labels
        self.test_dataset = test_dataset
        self.label_encoder = label_encoder
        self.seed = seed
        
        self.cluster_analysis = self._analyze_cluster_purity()
        
        self.model_names = self.label_encoder.classes_
        
    def _analyze_cluster_purity(self) -> Dict:
        """Analyze purity of each cluster."""
        unique_clusters = np.unique(self.cluster_labels)
        cluster_info = {}
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  #
                continue
                
            mask = self.cluster_labels == cluster_id
            cluster_true_labels = self.true_labels[mask]
            
            unique_models = np.unique(cluster_true_labels)
            model_counts = pd.Series(cluster_true_labels).value_counts()
            
            cluster_info[cluster_id] = {
                'size': np.sum(mask),
                'n_unique_models': len(unique_models),
                'most_common_model': model_counts.index[0],
                'most_common_count': model_counts.iloc[0],
                'purity': model_counts.iloc[0] / np.sum(mask),
                'model_distribution': model_counts.to_dict(),
                'is_pure': len(unique_models) == 1
            }
        
        return cluster_info
    
    def plot_embeddings_overview(
        self, 
        reduction_method: str = 'tsne',
        color_by: str = 'cluster', 
        figsize: Tuple[int, int] = (15, 10),
        alpha: float = 0.7,
        s: int = 50
    ):
        """Plot overview of embeddings in 2D space."""
        if reduction_method == 'tsne':
            if len(self.embeddings) < 4:
                reduction_method = 'pca'
                reducer = PCA(n_components=2, random_state=self.seed)
            else:
                reducer = TSNE(n_components=2, random_state=self.seed, perplexity=min(30, len(self.embeddings)//4))
        elif reduction_method == 'pca':
            reducer = PCA(n_components=2, random_state=self.seed)
        elif reduction_method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=self.seed)
        
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.cluster_labels, cmap='tab10',
            alpha=alpha, s=s
        )
        axes[0].set_title('Colored by Cluster Labels')
        axes[0].set_xlabel(f'{reduction_method.upper()} Component 1')
        axes[0].set_ylabel(f'{reduction_method.upper()} Component 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        scatter2 = axes[1].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=self.true_labels, cmap='tab20',
            alpha=alpha, s=s
        )
        axes[1].set_title('Colored by Ground Truth')
        axes[1].set_xlabel(f'{reduction_method.upper()} Component 1')
        axes[1].set_ylabel(f'{reduction_method.upper()} Component 2')
        plt.colorbar(scatter2, ax=axes[1])
        plt.tight_layout()
        plt.show()
        return embeddings_2d
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary of cluster purity analysis."""
        summary_data = []
        
        for cluster_id, info in self.cluster_analysis.items():
            model_name = self.model_names[info['most_common_model']]
            summary_data.append({
                'cluster_id': cluster_id,
                'size': info['size'],
                'n_unique_models': info['n_unique_models'],
                'most_common_model': model_name,
                'purity': info['purity'],
                'is_pure': info['is_pure']
            })
        
        df = pd.DataFrame(summary_data).sort_values('purity', ascending=False)
        return df
    
    def visualize_good_clusters(
        self, 
        n_clusters: int = 3, 
        images_per_cluster: int = 6,
        figsize_per_cluster: Tuple[int, int] = (15, 3)
    ):
        """Visualize pure clusters with representative images."""
        
        pure_clusters = [
            (cluster_id, info) for cluster_id, info in self.cluster_analysis.items() 
            if info['is_pure']
        ]
        pure_clusters.sort(key=lambda x: x[1]['size'], reverse=True) 
        
        if len(pure_clusters) == 0:
            logging.info("No pure clusters found!")
            return
        
        n_clusters = min(n_clusters, len(pure_clusters))

        fig, axes = plt.subplots(n_clusters, 1, figsize=(figsize_per_cluster[0], figsize_per_cluster[1] * n_clusters))
        if n_clusters == 1:
            axes = [axes]
        
        for i, (cluster_id, cluster_info) in enumerate(pure_clusters[:n_clusters]):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            n_to_show = min(images_per_cluster, len(cluster_indices))
            selected_indices = np.random.choice(cluster_indices, n_to_show, replace=False)
            
            model_name = self.model_names[cluster_info['most_common_model']]
            
            axes[i].set_title(
                f"Pure Cluster {cluster_id} - Model: {model_name} "
                f"(Size: {cluster_info['size']}, Purity: {cluster_info['purity']:.3f})"
            )
            axes[i].axis('off')
            
            for j, idx in enumerate(selected_indices):
                sample = self.test_dataset[idx]
                images = sample['images']
                
                if isinstance(images, list):
                    img = images[0]  
                else:
                    img = images
                
                if hasattr(img, 'numpy'):
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = np.array(img) / 255.0
                
                ax_img = plt.subplot2grid(
                    (n_clusters, images_per_cluster), (i, j),
                    fig=fig
                )
                ax_img.imshow(img_np)
                ax_img.axis('off')
                ax_img.set_title(f'Sample {j+1}', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_mixed_clusters(
        self, 
        n_clusters: int = 3,
        max_models_per_cluster: int = 4,
        figsize_per_cluster: Tuple[int, int] = (15, 3)
    ):
        """Visualize mixed clusters showing one image per different model."""
        
        mixed_clusters = [
            (cluster_id, info) for cluster_id, info in self.cluster_analysis.items() 
            if not info['is_pure'] and info['n_unique_models'] > 1
        ]
        mixed_clusters.sort(key=lambda x: x[1]['n_unique_models'], reverse=True)
        
        if len(mixed_clusters) == 0:
            logging.info("No mixed clusters found!")
            return
        
        n_clusters = min(n_clusters, len(mixed_clusters))

        fig, axes = plt.subplots(n_clusters, 1, figsize=(figsize_per_cluster[0], figsize_per_cluster[1] * n_clusters))
        if n_clusters == 1:
            axes = [axes]
        
        for i, (cluster_id, cluster_info) in enumerate(mixed_clusters[:n_clusters]):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_true_labels = self.true_labels[cluster_mask]
            
            unique_models = np.unique(cluster_true_labels)
            models_to_show = unique_models[:max_models_per_cluster]
            
            axes[i].set_title(
                f"Mixed Cluster {cluster_id} - {len(unique_models)} models "
                f"(Size: {cluster_info['size']}, Purity: {cluster_info['purity']:.3f})"
            )
            axes[i].axis('off')
            
            for j, model_label in enumerate(models_to_show):
                model_mask = cluster_true_labels == model_label
                local_indices = np.where(model_mask)[0]
                if len(local_indices) == 0:
                    continue
                
                global_idx = cluster_indices[local_indices[0]]
                sample = self.test_dataset[global_idx]
                images = sample['images']
                
                if isinstance(images, list):
                    img = images[0]  
                else:
                    img = images
                
                if hasattr(img, 'numpy'):
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = np.array(img) / 255.0
                
                ax_img = plt.subplot2grid(
                    (n_clusters, max_models_per_cluster), (i, j),
                    fig=fig
                )
                ax_img.imshow(img_np)
                ax_img.axis('off')
                
                model_name = self.model_names[model_label]
                count = cluster_info['model_distribution'].get(model_label, 0)
                ax_img.set_title(f'{model_name}\n({count} imgs)', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def print_cluster_statistics(self):
        """Print overall statistics about clustering quality."""
        total_clusters = len(self.cluster_analysis)
        pure_clusters = sum(1 for info in self.cluster_analysis.values() if info['is_pure'])
        mixed_clusters = total_clusters - pure_clusters
        logging.info("CLUSTERING STATISTICS")
        logging.info("=" * 40)
        logging.info(f"Total clusters: {total_clusters}")
        logging.info(f"Pure clusters: {pure_clusters} ({pure_clusters/total_clusters*100:.1f}%)")
        logging.info(f"Mixed clusters: {mixed_clusters} ({mixed_clusters/total_clusters*100:.1f}%)")
        
        if mixed_clusters > 0:
            avg_models_per_mixed = np.mean([
                info['n_unique_models'] for info in self.cluster_analysis.values() 
                if not info['is_pure']
            ])
            logging.info(f"Avg models per mixed cluster: {avg_models_per_mixed:.1f}")
        
        if mixed_clusters > 0:
            logging.info(f"\nMost mixed clusters:")
            mixed_info = [
                (cluster_id, info) for cluster_id, info in self.cluster_analysis.items()
                if not info['is_pure']
            ]
            mixed_info.sort(key=lambda x: x[1]['n_unique_models'], reverse=True)
            
            for cluster_id, info in mixed_info[:3]:
                logging.info(f"  Cluster {cluster_id}: {info['n_unique_models']} models, purity: {info['purity']:.3f}")