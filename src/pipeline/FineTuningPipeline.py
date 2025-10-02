"""
Pipeline de Fine-Tuning para modelos de visión.

Este módulo proporciona un pipeline enfocado exclusivamente en el fine-tuning
de modelos de visión.
"""

from __future__ import annotations

import logging
import shutil
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

from src.config.TransformConfig import create_standard_transform
from src.data.MyDataset import create_car_dataset
from src.models.Criterions import create_metric_learning_criterion
from src.models.MyVisionModel import create_vision_model
from src.utils.JsonUtils import safe_json_dump
from src.defaults import *

# Configuración de warnings y logging
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class FineTuningPipelineError(Exception):
    """Excepción personalizada para errores del pipeline de fine-tuning."""
    pass


class FineTuningPipeline:
    """
    Pipeline para fine-tuning de modelos de visión.
    
    Esta clase maneja todo el proceso de fine-tuning: creación del dataset,
    inicialización del modelo, entrenamiento, evaluación y guardado de resultados.
    
    Attributes:
        config: Configuración del pipeline.
        df: DataFrame con datos del dataset.
        dataset_dict: Diccionario con dataset, dataloaders y los sampler.
        model: Modelo de visión para fine-tuning.
        results: Diccionario con resultados del entrenamiento.
        
    Example:
        >>> pipeline = FineTuningPipeline(config_dict, dataframe)
        >>> results = pipeline.run_full_pipeline()
        >>> pipeline.save_results("experiment_name")
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        df: pd.DataFrame,
        experiment_name: Optional[str] = None
    ) -> None:
        """
        Inicializa el pipeline de fine-tuning.
        
        Args:
            config: Diccionario con configuración del pipeline.
            df: DataFrame con datos del dataset.
            experiment_name: Nombre del experimento (opcional).
        """
        self.config = config
        self.df = df
        self.experiment_name = experiment_name or f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Componentes del pipeline
        self.dataset_dict = None
        self.model = DEFAULT_MODEL_NAME
        self.results = {
            'config': config.copy(),
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat()
        }
        
        logging.info(f"Inicializado FineTuningPipeline: {self.experiment_name}")
    
    def create_dataset(self) -> None:
        """Crea el dataset para entrenamiento."""
        logging.info("Creando dataset...")
        
        # Crear transformaciones separadas para train y val/test
        augment = self.config.get('augment', DEFAULT_USE_AUGMENT)
        if augment:
             train_transform = create_standard_transform(
                size=tuple(self.config.get('image_size', DEFAULT_RESIZE)),
                grayscale=self.config.get('grayscale', DEFAULT_GRAYSCALE),
                use_bbox=self.config.get('use_bbox', DEFAULT_USE_BBOX),
                augment=augment
            )
        else: 
            train_transform = create_standard_transform(augment=augment)

        # Inference transform: SIN augmentación
        val_transform = create_standard_transform(
            size=tuple(self.config.get('image_size', [224, 224])),
            grayscale=self.config.get('grayscale', False),
            use_bbox=self.config.get('use_bbox', True),
            augment=False
        )

        # Determinar si usar IdentitySampler según el objetivo
        objective = self.config.get('objective', DEFAULT_OBJECTIVE)
        use_identity_sampler = (objective == 'metric_learning')
        
        self.dataset_dict = create_car_dataset(
            df=self.df,
            views=self.config.get('views', DEFAULT_VIEWS),
            min_images_for_abundant_class=self.config.get('min_images_for_abundant_class', DEFAULT_MIN_IMAGES_FOR_ABUNDANT_CLASS),
            seed=self.config.get('seed', DEFAULT_SEED),
            P=self.config.get('P', DEFAULT_P),
            K=self.config.get('K', DEFAULT_K),
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=self.config.get('batch_size', DEFAULT_BATCH_SIZE),
            num_workers=self.config.get('num_workers', DEFAULT_NUM_WORKERS),
            use_identity_sampler=use_identity_sampler,
            model_type=self.config.get('model_type', DEFAULT_MODEL_TYPE),
            description_include=self.config.get('description_include', DEFAULT_DESCRIPTION_INCLUDE)
        )
        
        logging.info(f"Dataset creado con objective='{objective}', use_identity_sampler={use_identity_sampler}")
        
        # Guardar estadísticas del dataset
        dataset_stats = self.dataset_dict['dataset'].get_dataset_statistics()
        self.results['dataset_stats'] = dataset_stats
    
    def create_model(self) -> None:
        """Crea e inicializa el modelo."""
        if self.dataset_dict is None:
            raise FineTuningPipelineError("Dataset no creado. Ejecutar create_dataset() primero.")
        
        logging.info("Creando modelo...")  
        device = torch.device(self.config.get('device', 'cpu'))
        
        self.model = create_vision_model(
            name=f"{self.config.get('model_name', DEFAULT_MODEL_NAME)}_{self.experiment_name}",
            model_name=self.config.get('model_name', DEFAULT_MODEL_NAME),
            weights=self.config.get('weights', DEFAULT_WEIGHTS),
            device=device,
            objective=self.config.get('objective', DEFAULT_OBJECTIVE),
            dataset_dict=self.dataset_dict,
            batch_size=self.config.get('batch_size', DEFAULT_BATCH_SIZE),
            num_workers=self.config.get('num_workers', DEFAULT_NUM_WORKERS),
            pin_memory=self.config.get('pin_memory', DEFAULT_PIN_MEMORY)
        )
        
        # Guardar información del modelo
        model_info = self.model.get_model_info()
        self.results['model_info'] = model_info
    
    def extract_baseline_embeddings(self) -> torch.Tensor:
        """Extrae embeddings antes del fine-tuning."""
        if self.model is None:
            raise FineTuningPipelineError("Modelo no creado. Ejecutar create_model() primero.")
        use_arcface = self.config.get('finetune_criterion') == 'ArcFaceLoss'

        logging.info("Extrayendo embeddings baseline...")

        baseline_embeddings = self.model.extract_val_embeddings(use_arcface=use_arcface)
        baseline_eval = self.model.evaluate()
        
        # Guardar resultados baseline
        self.results['baseline'] = {
            'embeddings_shape': list(baseline_embeddings.shape),
            'evaluation': baseline_eval
        }
        # Guardar embeddings para uso posterior
        self.results['baseline_embeddings'] = baseline_embeddings
        
        logging.info(f"Embeddings baseline extraídos: {baseline_embeddings.shape}")
        
        # Log según el tipo de métrica
        if 'accuracy' in baseline_eval:
            logging.info(f"Accuracy baseline: {baseline_eval['accuracy']:.4f}")
        elif 'recall@1' in baseline_eval:
            logging.info(f"Recall@1 baseline: {baseline_eval['recall@1']:.4f}")
            logging.info(f"Recall@5 baseline: {baseline_eval['recall@5']:.4f}")
        
        return baseline_embeddings
    
    def run_finetuning(self) -> Dict[str, Any]:
        """Ejecuta el fine-tuning del modelo."""
        if self.model is None:
            raise FineTuningPipelineError("Modelo no creado. Ejecutar create_model() primero.")
                
        # Configurar criterio de pérdida desde config
        criterion_name = self.config.get('finetune_criterion', DEFAULT_FINETUNE_CRITERION)
        if criterion_name in ['TripletLoss', 'ContrastiveLoss', 'ArcFaceLoss']:
            # Validar que el objetivo sea metric_learning
            if self.model.objective != 'metric_learning':
                raise FineTuningPipelineError(
                    f"Criterio '{criterion_name}' requiere objective='metric_learning', "
                    f"pero el modelo tiene objective='{self.model.objective}'"
                )
            criterion = create_metric_learning_criterion(
                loss_type=criterion_name, 
                embedding_dim=self.model.embedding_dim, 
                num_classes=self.dataset_dict['dataset'].num_models
            )
        else:
            # Validar que el objetivo sea classification para criterios estándar
            if self.model.objective != 'classification':
                logging.warning(
                    f"Usando criterio '{criterion_name}' con objective='{self.model.objective}'. "
                    "Considere usar un criterio de metric learning apropiado."
                )
            criterion_cls = getattr(torch.nn, criterion_name)
            criterion = criterion_cls()
        
        # Mover el criterio al dispositivo del modelo
        criterion.to(self.model.device)
        
        # Configurar optimizador desde config
        optimizer_type = self.config.get('finetune_optimizer_type', DEFAULT_FINETUNE_OPTIMIZER_TYPE)
        base_lr = self.config.get('finetune_backbone_lr', DEFAULT_BACKBONE_LR)
        head_lr = self.config.get('finetune_head_lr', DEFAULT_HEAD_LR)
        weight_decay = self.config.get('finetune_optimizer_weight_decay', DEFAULT_WEIGHT_DECAY)
        
        optimizer_cls = getattr(torch.optim, optimizer_type)
        optimizer_params = [
            {"params": self.model.model.parameters(), "lr": base_lr, "weight_decay": weight_decay},
            {"params": self.model.head_layer.parameters(), "lr": head_lr, "weight_decay": weight_decay}
        ]
        
        # Agregar parámetros de arcface_layer si existe
        if self.model.arcface_layer is not None:
            optimizer_params.append({
                "params": self.model.arcface_layer.parameters(), 
                "lr": head_lr, 
                "weight_decay": weight_decay
            })
            logging.info("Parámetros de arcface_layer agregados al optimizador")
        
        optimizer = optimizer_cls(optimizer_params)
        
        # Configurar scheduler
        scheduler = None
        if self.config.get('use_scheduler', DEFAULT_USE_SCHEDULER):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.config.get('finetune_epochs', 10)
            )
        
        # Configurar early stopping
        early_stopping = None
        if self.config.get('use_early_stopping', DEFAULT_USE_EARLY_STOPPING):
            early_stopping = {'patience': self.config.get('patience', DEFAULT_PATIENCE)}
        
        # Ejecutar fine-tuning
        training_history = self.model.finetune(
            criterion=criterion,
            optimizer=optimizer,
            epochs=self.config.get('finetune_epochs', 10),
            warmup_epochs=self.config.get('warmup_epochs', 1),
            scheduler=scheduler,
            early_stopping=early_stopping
        )
        
        # Guardar resultados del fine-tuning
        self.results['finetuning'] = {
            'training_history': training_history,
            'final_train_loss': training_history['train_loss'][-1],
            'final_val_loss': training_history['val_loss'][-1],
            'final_val_accuracy': training_history['val_accuracy'][-1],
            'best_val_accuracy': max(training_history['val_accuracy']),
            'total_epochs': len(training_history['train_loss'])
        }
        
        return training_history
    
    def extract_finetuned_embeddings(self) -> torch.Tensor:
        """Extrae embeddings después del fine-tuning."""
        if self.model is None:
            raise FineTuningPipelineError("Modelo no creado.")
        
        logging.info("Extrayendo embeddings post fine-tuning...")
        
        finetuned_embeddings = self.model.extract_val_embeddings()
        finetuned_eval = self.model.evaluate()
        
        # Guardar resultados post fine-tuning
        self.results['finetuned'] = {
            'embeddings_shape': list(finetuned_embeddings.shape),
            'evaluation': finetuned_eval
        }
        # Guardar embeddings para uso posterior
        self.results['finetuned_embeddings'] = finetuned_embeddings
        
        logging.info(f"Embeddings post fine-tuning extraídos: {finetuned_embeddings.shape}")
        
        # Log según el tipo de métrica
        if 'accuracy' in finetuned_eval:
            logging.info(f"Accuracy post fine-tuning: {finetuned_eval['accuracy']:.4f}")
        elif 'recall@1' in finetuned_eval:
            logging.info(f"Recall@1 post fine-tuning: {finetuned_eval['recall@1']:.4f}")
            logging.info(f"Recall@5 post fine-tuning: {finetuned_eval['recall@5']:.4f}")
        
        return finetuned_embeddings
    
    def save_results(self, save_dir: Union[str, Path] = "results") -> Path:
        """
        Guarda todos los resultados en un directorio y crea un ZIP.
        
        Args:
            save_dir: Directorio base para guardar resultados.
            
        Returns:
            Path al archivo ZIP creado.
        """
        save_dir = Path(save_dir)
        experiment_dir = save_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Guardando resultados en: {experiment_dir}")
        
        # Guardar configuración y resultados (sin embeddings que son muy grandes para JSON)
        safe_json_dump(self.config, experiment_dir / "config.json")
        
        # Finalizar resultados con tiempo de finalización
        self.results['end_time'] = datetime.now().isoformat()
        
        # Separar embeddings y historial del JSON
        results_for_json = self.results.copy()
        baseline_embeddings = None
        finetuned_embeddings = None
        training_history = None
        
        if 'baseline_embeddings' in results_for_json:
            baseline_embeddings = results_for_json.pop('baseline_embeddings')
        if 'finetuned_embeddings' in results_for_json:
            finetuned_embeddings = results_for_json.pop('finetuned_embeddings')
        if 'finetuning' in results_for_json and 'training_history' in results_for_json['finetuning']:
            training_history = results_for_json['finetuning']['training_history']
        
        safe_json_dump(results_for_json, experiment_dir / "results.json")
        
        # Guardar embeddings como tensors de PyTorch
        if baseline_embeddings is not None:
            torch.save(baseline_embeddings, experiment_dir / "baseline_embeddings.pt")
            logging.info(f"Baseline embeddings guardados: {baseline_embeddings.shape}")
            
        if finetuned_embeddings is not None:
            torch.save(finetuned_embeddings, experiment_dir / "finetuned_embeddings.pt")
            logging.info(f"Fine-tuned embeddings guardados: {finetuned_embeddings.shape}")
        
        # Guardar historial de entrenamiento como tensor/pickle
        if training_history is not None:
            torch.save(training_history, experiment_dir / "training_history.pt")
            logging.info("Historial de entrenamiento guardado")
        
        # Guardar modelo si existe
        if self.model is not None:
            model_path = experiment_dir / "model.pth"
            self.model.save_model(model_path)
            logging.info(f"Modelo guardado: {model_path}")
        
        # Guardar estadísticas del dataset
        if 'dataset_stats' in self.results:
            stats_df = pd.DataFrame([self.results['dataset_stats']['overview']])
            stats_df.to_csv(experiment_dir / "dataset_stats.csv", index=False)
        
        # Crear archivo ZIP
        zip_path = save_dir / f"{self.experiment_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in experiment_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(save_dir)
                    zipf.write(file_path, arcname)
        
        # Limpiar directorio temporal (opcional)
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        
        logging.info(f"Resultados guardados en ZIP: {zip_path}")
        return zip_path
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de fine-tuning.
        
        Returns:
            Diccionario con todos los resultados.
        """
        logging.info(f"=== INICIANDO PIPELINE COMPLETO: {self.experiment_name} ===")
        
        try:
            # 1. Crear dataset
            self.create_dataset()
            
            # 2. Crear modelo
            self.create_model()
            
            # 3. Extraer embeddings baseline
            self.extract_baseline_embeddings()
            
            # 4. Fine-tuning
            self.run_finetuning()
            
            # 5. Extraer embeddings post fine-tuning
            self.extract_finetuned_embeddings()
            
            # 6. Calcular mejora según el tipo de métrica
            baseline_eval = self.results['baseline']['evaluation']
            finetuned_eval = self.results['finetuned']['evaluation']
            
            if 'accuracy' in baseline_eval:
                # Modo classification
                baseline_metric = baseline_eval['accuracy']
                finetuned_metric = finetuned_eval['accuracy']
                metric_name = 'accuracy'
            elif 'recall@1' in baseline_eval:
                # Modo metric_learning
                baseline_metric = baseline_eval['recall@1']
                finetuned_metric = finetuned_eval['recall@1']
                metric_name = 'recall@1'
            else:
                raise FineTuningPipelineError("No se encontró métrica válida en los resultados de evaluación")
            
            improvement = finetuned_metric - baseline_metric
            improvement_percentage = (improvement / baseline_metric) * 100 if baseline_metric > 0 else 0.0
            
            self.results['summary'] = {
                'metric_name': metric_name,
                f'baseline_{metric_name}': baseline_metric,
                f'finetuned_{metric_name}': finetuned_metric,
                f'{metric_name}_improvement': improvement,
                'improvement_percentage': improvement_percentage,
                'objective': self.model.objective
            }
            
            # Agregar métricas adicionales para metric_learning
            if 'recall@5' in baseline_eval:
                self.results['summary']['baseline_recall@5'] = baseline_eval['recall@5']
                self.results['summary']['finetuned_recall@5'] = finetuned_eval['recall@5']
                self.results['summary']['recall@5_improvement'] = finetuned_eval['recall@5'] - baseline_eval['recall@5']
            
            logging.info(f"=== PIPELINE COMPLETADO ===")
            logging.info(f"Mejora de {metric_name}: {improvement:.4f} ({improvement_percentage:.2f}%)")
            
            # 7. Guardar resultados automáticamente
            zip_path = self.save_results()
            self.results['saved_to'] = str(zip_path)
            
            logging.info(f"Resultados guardados en: {zip_path}")
            
            return self.results
            
        except Exception as e:
            logging.error(f"Error en pipeline: {e}")
            self.results['error'] = str(e)
            raise FineTuningPipelineError(f"Error ejecutando pipeline: {e}") from e
    
    def __str__(self) -> str:
        """Representación string del pipeline."""
        return f"FineTuningPipeline(experiment={self.experiment_name})"
    
    def __repr__(self) -> str:
        """Representación detallada del pipeline."""
        return (
            f"FineTuningPipeline("
            f"experiment={self.experiment_name}, "
            f"dataset={'✓' if self.dataset_dict else '✗'}, "
            f"model={'✓' if self.model else '✗'})"
        )


# Función de conveniencia
def create_finetuning_pipeline(
    config: Dict[str, Any],
    df: pd.DataFrame,
    experiment_name: Optional[str] = None
) -> FineTuningPipeline:
    """
    Función de conveniencia para crear un pipeline de fine-tuning.
    
    Args:
        config: Configuración del pipeline.
        df: DataFrame con datos.
        experiment_name: Nombre del experimento.
        
    Returns:
        Pipeline de fine-tuning configurado.
    """
    return FineTuningPipeline(config, df, experiment_name)