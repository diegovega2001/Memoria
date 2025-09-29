"""
Pipeline de análisis de embeddings para modelos de visión.

Este módulo proporciona un pipeline enfocado en el análisis y visualización
de embeddings, incluyendo reducción de dimensionalidad y clustering.
"""

from __future__ import annotations

import json
import logging
import shutil
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch


from src.utils.JsonUtils import safe_json_dump

from src.config.TransformConfig import create_standard_transform
from src.data.MyDataset import create_car_dataset
from src.utils.ClusteringAnalyzer import ClusteringAnalyzer
from src.utils.ClusterVisualizer import ClusterVisualizer
from src.utils.DimensionalityReducer import DimensionalityReducer
from src.utils.JsonUtils import safe_json_dump

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class EmbeddingsPipelineError(Exception):
    """Excepción personalizada para errores del pipeline de embeddings."""
    pass


class EmbeddingsPipeline:
    """
    Pipeline para análisis de embeddings de modelos de visión.
    
    Esta clase maneja el análisis completo de embeddings: extracción,
    reducción de dimensionalidad, clustering y visualización.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        df: pd.DataFrame,
        experiment_name: Optional[str] = None
    ) -> None:
        self.config = config
        self.df = df
        self.experiment_name = experiment_name or f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Componentes del pipeline
        self.dataset_dict = None
        self.baseline_embeddings = None
        self.baseline_labels = None
        self.finetuned_embeddings = None
        
        # Información de los embeddings cargados
        self.baseline_info = None
        self.finetuned_info = None
        
        self.results = {
            'config': config.copy(),
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat()
        }
        
        logging.info(f"Inicializado EmbeddingsPipeline: {self.experiment_name}")
    
    def create_dataset_for_labels(self) -> None:
        """Crea el dataset solo para obtener las etiquetas correctas."""
        logging.info("Creando dataset para obtener etiquetas...")
        
        transform = create_standard_transform(
            size=tuple(self.config.get('image_size', [224, 224])),
            grayscale=self.config.get('grayscale', False),
            use_bbox=self.config.get('use_bbox', True)
        )
        
        self.dataset_dict = create_car_dataset(
            df=self.df,
            views=self.config.get('views', ['front']),
            min_images_for_abundant_class=self.config.get('min_images_for_abundant_class', 5),
            seed=self.config.get('seed', 3),
            transform=transform,
            augment=False,
            model_type='vision',
            description_include=''
        )

        self.dataset_dict['dataset'].set_split('val')

        labels = []
        for sample_tuple in self.dataset_dict['dataset'].val_samples:
            model_year_tuple = sample_tuple[0]
            label_str = f"{model_year_tuple[0]}_{model_year_tuple[1]}"
            label = self.dataset_dict['dataset'].label_encoder.transform([label_str])[0]
            labels.append(label)
        
        self.baseline_labels = torch.tensor(labels)
        
        self.results['dataset_info'] = {
            'num_models': self.dataset_dict['dataset'].num_models,
            'val_samples': len(self.dataset_dict['dataset'].val_samples),
            'views': self.dataset_dict['dataset'].views
        }
        
        logging.info(f"Dataset creado - val samples: {len(self.dataset_dict['dataset'].val_samples)}")
        logging.info(f"Etiquetas extraídas: {len(self.baseline_labels)} muestras")
    
    def load_embeddings_from_files(
        self, 
        baseline_embeddings_path: Union[str, Path],
        finetuned_embeddings_path: Optional[Union[str, Path]] = None
    ) -> None:
        logging.info("Cargando embeddings desde archivos...")
        
        baseline_path = Path(baseline_embeddings_path)
        if not baseline_path.exists():
            raise EmbeddingsPipelineError(f"Archivo de embeddings baseline no encontrado: {baseline_path}")
        
        self.baseline_embeddings = torch.load(baseline_path, map_location='cpu')
        self.baseline_info = {
            'path': str(baseline_path),
            'shape': list(self.baseline_embeddings.shape),
            'device': str(self.baseline_embeddings.device)
        }
        
        logging.info(f"Embeddings baseline cargados: {self.baseline_embeddings.shape}")
        
        if finetuned_embeddings_path:
            finetuned_path = Path(finetuned_embeddings_path)
            if not finetuned_path.exists():
                raise EmbeddingsPipelineError(f"Archivo de embeddings fine-tuned no encontrado: {finetuned_path}")
            
            self.finetuned_embeddings = torch.load(finetuned_path, map_location='cpu')
            self.finetuned_info = {
                'path': str(finetuned_path),
                'shape': list(self.finetuned_embeddings.shape),
                'device': str(self.finetuned_embeddings.device)
            }
            
            logging.info(f"Embeddings fine-tuned cargados: {self.finetuned_embeddings.shape}")
            
            if self.baseline_embeddings.shape != self.finetuned_embeddings.shape:
                raise EmbeddingsPipelineError(
                    f"Las formas de embeddings no coinciden: "
                    f"baseline {self.baseline_embeddings.shape} vs "
                    f"finetuned {self.finetuned_embeddings.shape}"
                )
        
        self.results['embeddings_info'] = {
            'baseline': self.baseline_info,
            'finetuned': self.finetuned_info if finetuned_embeddings_path else None
        }
    
    def load_embeddings_from_zip(
        self,
        zip_path: Union[str, Path]
    ) -> None:
        import tempfile
        
        logging.info(f"Cargando embeddings desde ZIP: {zip_path}")
        
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise EmbeddingsPipelineError(f"Archivo ZIP no encontrado: {zip_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            baseline_files = list(temp_path.rglob("baseline_embeddings.pt"))
            finetuned_files = list(temp_path.rglob("finetuned_embeddings.pt"))
            
            if not baseline_files:
                raise EmbeddingsPipelineError("No se encontró baseline_embeddings.pt en el ZIP")
            
            baseline_path = baseline_files[0]
            finetuned_path = finetuned_files[0] if finetuned_files else None
            
            self.load_embeddings_from_files(baseline_path, finetuned_path)
            
            config_files = list(temp_path.rglob("config.json"))
            if config_files:
                with open(config_files[0], 'r') as f:
                    loaded_config = json.load(f)
                    self.results['original_config'] = loaded_config
                    logging.info("Configuración original cargada desde ZIP")
    
    def analyze_baseline_embeddings(self) -> Dict[str, Any]:
        if self.baseline_embeddings is None:
            raise EmbeddingsPipelineError("Embeddings baseline no cargados. Ejecutar load_embeddings_from_files() primero.")
        
        if self.baseline_labels is None:
            raise EmbeddingsPipelineError("Etiquetas no disponibles. Ejecutar create_dataset_for_labels() primero.")
        
        logging.info("Analizando embeddings baseline cargados...")
        
        baseline_results = self._analyze_embeddings(
            embeddings=self.baseline_embeddings,
            labels=self.baseline_labels,
            phase='baseline'
        )
        
        self.results['baseline_analysis'] = baseline_results
        
        logging.info("Análisis baseline completado!")
        return baseline_results
    
    def analyze_finetuned_embeddings(self) -> Dict[str, Any]:
        if self.finetuned_embeddings is None:
            raise EmbeddingsPipelineError("Embeddings fine-tuned no cargados. Deben cargarse con load_embeddings_from_files().")
        
        if self.baseline_labels is None:
            raise EmbeddingsPipelineError("Etiquetas no disponibles. Ejecutar create_dataset_for_labels() primero.")
        
        logging.info("Analizando embeddings fine-tuneados cargados...")
        
        finetuned_results = self._analyze_embeddings(
            embeddings=self.finetuned_embeddings,
            labels=self.baseline_labels,
            phase='finetuned'
        )
        
        self.results['finetuned_analysis'] = finetuned_results
        
        logging.info("Análisis fine-tuneado completado!")
        return finetuned_results
    
    def _analyze_embeddings(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor, 
        phase: str
    ) -> Dict[str, Any]:
        results = {}
        
        logging.info(f"Ejecutando reducción de dimensionalidad - {phase}...")
        
        reducer = DimensionalityReducer(
            embeddings=embeddings,
            labels=labels,
            seed=self.config.get('seed', 3),
            optimizer_trials=self.config.get('reducer_trials', 50),
            available_methods=self.config.get('reducer_methods', ['pca', 'umap']),
            n_jobs=self.config.get('n_jobs', -1),
            use_incremental=self.config.get('use_incremental', True)
        )
        
        reduction_scores = reducer.reduce_all()
        best_method, best_embeddings = reducer.get_best_result()
        
        results['reduction'] = {
            'scores': reduction_scores,
            'best_method': best_method,
            'best_embeddings_shape': list(best_embeddings.shape)
        }
        
        logging.info(f"Ejecutando clustering - {phase}...")
        
        clustering = ClusteringAnalyzer(
            embeddings=best_embeddings,
            true_labels=labels,
            seed=self.config.get('seed', 3),
            optimizer_trials=self.config.get('clustering_trials', 50),
            available_methods=self.config.get('clustering_methods', ['dbscan', 'hdbscan']),
            n_jobs=self.config.get('n_jobs', -1)
        )
        
        clustering_results = clustering.cluster_all()
        logging.info(f"Clustering results keys: {list(clustering_results.keys()) if clustering_results else 'Empty results'}")
        
        if not clustering_results:
            raise ValueError(f"No se pudieron generar resultados de clustering para {phase}")
            
        comparison_df = clustering.compare_methods()
        best_clustering_method, best_cluster_labels = clustering.get_best_result()
        
        results['clustering'] = {
            'results': clustering_results,
            'comparison_df': comparison_df.to_dict('records'),
            'best_method': best_clustering_method
        }
        
        if self.config.get('generate_visualizations', True):
            logging.info(f"Generando visualizaciones - {phase}...")
            
            visualizer = ClusterVisualizer(
                embeddings=best_embeddings,
                cluster_labels=best_cluster_labels,
                true_labels=labels,
                val_samples=self.dataset_dict['dataset'].val_samples,
                label_encoder=self.dataset_dict['dataset'].label_encoder,
                seed=self.config.get('seed', 3)
            )
            
            # Estadísticas y resumen
            visualizer.print_cluster_statistics()
            summary_df = clustering.get_cluster_summary(best_clustering_method)

            # Número de clusters a visualizar 
            n_to_vis = int(self.config.get('n_clusters_to_visualize', 3))
            max_classes_per_cluster = int(self.config.get('max_classes_per_cluster_viz', 8))

            # Usar estrategia adaptativa de visualización
            try:
                visualizer.visualize_best_available_clusters(n_to_vis, max_classes_per_cluster)
            except Exception as e:
                logging.error(f"Error en visualización adaptativa de clusters: {e}")
                # Fallback: intentar visualizaciones individuales
                try:
                    logging.info("Intentando visualización de clusters puros como fallback...")
                    visualizer.visualize_good_clusters(n_to_vis, max_classes_per_cluster)
                except Exception as e2:
                    logging.warning(f"Error visualizando clusters puros: {e2}")

                try:
                    logging.info("Intentando visualización de clusters mixtos como fallback...")
                    visualizer.visualize_mixed_clusters(n_to_vis, max_classes_per_cluster)
                except Exception as e3:
                    logging.warning(f"Error visualizando clusters mixtos: {e3}")

            # Obtener solapamiento de clases entre clusters
            try:
                overlap_df = visualizer.get_class_cluster_overlap()
                overlap_records = overlap_df.to_dict('records') if isinstance(overlap_df, pd.DataFrame) else []
            except Exception as e:
                logging.warning(f"Error calculando solapamiento de clases: {e}")
                overlap_records = []

            # Obtener overlap de clases
            try:
                overlap_df = visualizer.get_class_cluster_overlap()
                overlap_records = overlap_df.to_dict('records') if not overlap_df.empty else []
            except Exception as e:
                logging.warning(f"No se pudo generar overlap de clases: {e}")
                overlap_records = []
            
            results['visualization'] = {
                'cluster_summary': summary_df.to_dict('records'),
                'cluster_analysis': visualizer.cluster_analysis,
                'class_cluster_overlap': overlap_records
            }
        
        logging.info(f"Análisis {phase} - Mejor reducción: {best_method}, Mejor clustering: {best_clustering_method}")
        
        return results
    
    def compare_results(self) -> Dict[str, Any]:
        if 'baseline_analysis' not in self.results or 'finetuned_analysis' not in self.results:
            raise EmbeddingsPipelineError("Ambos análisis (baseline y finetuned) deben completarse primero.")
        
        logging.info("Comparando resultados baseline vs fine-tuned...")
        
        baseline = self.results['baseline_analysis']
        finetuned = self.results['finetuned_analysis']
        
        baseline_clustering = baseline['clustering']['comparison_df']
        finetuned_clustering = finetuned['clustering']['comparison_df']
        
        comparison = {
            'reduction_methods': {
                'baseline_best': baseline['reduction']['best_method'],
                'finetuned_best': finetuned['reduction']['best_method']
            },
            'clustering_methods': {
                'baseline_best': baseline['clustering']['best_method'],
                'finetuned_best': finetuned['clustering']['best_method']
            },
            'clustering_metrics_comparison': self._compare_clustering_metrics(
                baseline_clustering, finetuned_clustering
            )
        }
        
        if baseline_clustering and finetuned_clustering:
            baseline_best = max(baseline_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
            finetuned_best = max(finetuned_clustering, key=lambda x: x.get('adjusted_rand_score', 0))
            
            comparison['performance_improvement'] = {
                'baseline_ari': baseline_best.get('adjusted_rand_score', 0),
                'finetuned_ari': finetuned_best.get('adjusted_rand_score', 0),
                'ari_improvement': finetuned_best.get('adjusted_rand_score', 0) - baseline_best.get('adjusted_rand_score', 0)
            }
        
        self.results['comparison'] = comparison
        
        logging.info("Comparación completada!")
        logging.info(f"Mejor método baseline: {comparison['clustering_methods']['baseline_best']}")
        logging.info(f"Mejor método fine-tuned: {comparison['clustering_methods']['finetuned_best']}")
        
        return comparison
    
    def _compare_clustering_metrics(
        self, 
        baseline_results: list, 
        finetuned_results: list
    ) -> Dict[str, Any]:
        if not baseline_results or not finetuned_results:
            return {}
        
        metrics = ['adjusted_rand_score', 'silhouette_score', 'calinski_harabasz_score']
        comparison = {}
        
        for metric in metrics:
            baseline_values = [r.get(metric, 0) for r in baseline_results]
            finetuned_values = [r.get(metric, 0) for r in finetuned_results]
            
            if baseline_values and finetuned_values:
                comparison[metric] = {
                    'baseline_max': max(baseline_values),
                    'baseline_mean': np.mean(baseline_values),
                    'finetuned_max': max(finetuned_values),
                    'finetuned_mean': np.mean(finetuned_values),
                    'improvement_max': max(finetuned_values) - max(baseline_values),
                    'improvement_mean': np.mean(finetuned_values) - np.mean(baseline_values)
                }
        
        return comparison
    
    def save_results(self, save_dir: Union[str, Path] = "results") -> Path:
        save_dir = Path(save_dir)
        experiment_dir = save_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Guardando resultados en: {experiment_dir}")
        
        safe_json_dump(self.config, experiment_dir / "config.json")
        
        self.results['end_time'] = datetime.now().isoformat()
        
        results_to_save = self.results.copy()
        
        safe_json_dump(results_to_save, experiment_dir / "results.json")
        
        if self.baseline_embeddings is not None:
            np.save(experiment_dir / "baseline_embeddings.npy", self.baseline_embeddings.numpy())
        
        if self.finetuned_embeddings is not None:
            np.save(experiment_dir / "finetuned_embeddings.npy", self.finetuned_embeddings.numpy())
        
        if self.baseline_labels is not None:
            np.save(experiment_dir / "labels.npy", self.baseline_labels.numpy())
        
        if 'comparison' in self.results:
            comparison_df = pd.DataFrame([self.results['comparison']['performance_improvement']])
            comparison_df.to_csv(experiment_dir / "performance_comparison.csv", index=False)
        
        zip_path = save_dir / f"{self.experiment_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in experiment_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(save_dir)
                    zipf.write(file_path, arcname)
        
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        
        logging.info(f"Resultados guardados en ZIP: {zip_path}")
        return zip_path
    
    def run_full_analysis_from_zip(self, zip_path: Union[str, Path]) -> Dict[str, Any]:
        logging.info(f"=== INICIANDO ANÁLISIS DESDE ZIP: {self.experiment_name} ===")
        
        try:
            self.create_dataset_for_labels()
            self.load_embeddings_from_zip(zip_path)
            self.analyze_baseline_embeddings()
            
            if self.finetuned_embeddings is not None:
                self.analyze_finetuned_embeddings()
                self.compare_results()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info(f"=== ANÁLISIS COMPLETO COMPLETADO ===")
            logging.info(f"Resultados guardados en: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error ejecutando análisis: {e}") from e
    
    def run_full_analysis_from_files(
        self,
        baseline_embeddings_path: Union[str, Path],
        finetuned_embeddings_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        logging.info(f"=== INICIANDO ANÁLISIS DESDE ARCHIVOS: {self.experiment_name} ===")
        
        try:
            self.create_dataset_for_labels()
            self.load_embeddings_from_files(baseline_embeddings_path, finetuned_embeddings_path)
            self.analyze_baseline_embeddings()
            
            if self.finetuned_embeddings is not None:
                self.analyze_finetuned_embeddings()
                self.compare_results()
            
            zip_result_path = self.save_results()
            self.results['saved_to'] = str(zip_result_path)
            
            logging.info(f"=== ANÁLISIS COMPLETO COMPLETADO ===")
            logging.info(f"Resultados guardados en: {zip_result_path}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error en análisis: {e}")
            self.results['error'] = str(e)
            raise EmbeddingsPipelineError(f"Error ejecutando análisis: {e}") from e
    
    def __str__(self) -> str:
        return f"EmbeddingsPipeline(experiment={self.experiment_name})"
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingsPipeline("
            f"experiment={self.experiment_name}, "
            f"dataset={'✓' if self.dataset_dict else '✗'}, "
            f"baseline={'✓' if self.baseline_embeddings is not None else '✗'}, "
            f"finetuned={'✓' if self.finetuned_embeddings is not None else '✗'})"
        )


# Función de conveniencia
def create_embeddings_pipeline(
    config: Dict[str, Any],
    df: pd.DataFrame,
    experiment_name: Optional[str] = None
) -> EmbeddingsPipeline:
    return EmbeddingsPipeline(config, df, experiment_name)