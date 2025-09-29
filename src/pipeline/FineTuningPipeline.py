"""
Pipeline de Fine-Tuning para modelos de visión.

Este módulo proporciona un pipeline enfocado exclusivamente en el fine-tuning
de modelos de visión, incluyendo entrenamiento, evaluación y guardado de resultados.
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
from torch.utils.data import DataLoader

from src.config.TransformConfig import create_standard_transform
from src.utils.JsonUtils import safe_json_dump
from src.data.MyDataset import create_car_dataset
from src.models.MyVisionModel import create_vision_model

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
        dataset_dict: Diccionario con dataset, dataloaders y test sampler.
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
        self.model = None
        self.results = {
            'config': config.copy(),
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat()
        }
        
        logging.info(f"Inicializado FineTuningPipeline: {self.experiment_name}")
    
    def create_dataset(self) -> None:
        """Crea el dataset para entrenamiento."""
        logging.info("Creando dataset...")
        
        # Crear transformaciones con bbox si está habilitado
        transform = create_standard_transform(
            size=tuple(self.config.get('image_size', [224, 224])),
            grayscale=self.config.get('grayscale', False),
            use_bbox=self.config.get('use_bbox', True)
        )
        
        # Crear dataset
        self.dataset_dict = create_car_dataset(
            df=self.df,
            views=self.config.get('views', ['front']),
            min_images_for_abundant_class=self.config.get('min_images_for_abundant_class', 6),
            seed=self.config.get('seed', 3),
            transform=transform,
            augment=self.config.get('augment', False),
            model_type='vision',
            description_include=''
        )
        
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
            name=f"{self.config.get('model_name', 'resnet50')}_{self.experiment_name}",
            model_name=self.config.get('model_name', 'resnet50'),
            weights=self.config.get('weights', 'IMAGENET1K_V1'),
            device=device,
            dataset_dict=self.dataset_dict,
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', True)
        )
        
        # Guardar información del modelo
        model_info = self.model.get_model_info()
        self.results['model_info'] = model_info
    
    def extract_baseline_embeddings(self) -> torch.Tensor:
        """Extrae embeddings antes del fine-tuning."""
        if self.model is None:
            raise FineTuningPipelineError("Modelo no creado. Ejecutar create_model() primero.")
        
        logging.info("Extrayendo embeddings baseline...")
        
        baseline_embeddings = self.model.extract_test_embeddings()
        baseline_eval = self.model.evaluate()
        
        # Guardar resultados baseline
        self.results['baseline'] = {
            'embeddings_shape': list(baseline_embeddings.shape),
            'evaluation': baseline_eval
        }
        # Guardar embeddings para uso posterior
        self.results['baseline_embeddings'] = baseline_embeddings
        
        logging.info(f"Embeddings baseline extraídos: {baseline_embeddings.shape}")
        logging.info(f"Accuracy baseline: {baseline_eval['accuracy']:.4f}")
        
        return baseline_embeddings
    
    def run_finetuning(self) -> Dict[str, Any]:
        """Ejecuta el fine-tuning del modelo."""
        if self.model is None:
            raise FineTuningPipelineError("Modelo no creado. Ejecutar create_model() primero.")
                
        # Configurar criterio de pérdida desde config
        criterion_name = self.config.get('finetune criterion', 'CrossEntropyLoss')
        criterion_cls = getattr(torch.nn, criterion_name)
        criterion = criterion_cls()
        
        # Configurar optimizador desde config
        optimizer_type = self.config.get('finetune optimizer type', 'AdamW')
        base_lr = self.config.get('finetune optimizer lr', 1e-4)
        head_lr = self.config.get('finetune optimizer head_lr', base_lr)
        weight_decay = self.config.get('finetune optimizer weight_decay', 1e-6)
        
        optimizer_cls = getattr(torch.optim, optimizer_type)
        optimizer_params = [
            {"params": self.model.model.parameters(), "lr": base_lr, "weight_decay": weight_decay},
            {"params": self.model.classification_layer.parameters(), "lr": head_lr, "weight_decay": weight_decay}
        ]
        optimizer = optimizer_cls(optimizer_params)
        
        # Configurar scheduler
        scheduler = None
        if self.config.get('use_scheduler', True):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.config.get('finetune epochs', 10)
            )
        
        # Configurar early stopping
        early_stopping = None
        if self.config.get('use_early_stopping', True):
            early_stopping = {'patience': self.config.get('patience', 5)}
        
        # Ejecutar fine-tuning
        training_history = self.model.finetune(
            criterion=criterion,
            optimizer=optimizer,
            epochs=self.config.get('finetune epochs', 10),
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
        
        finetuned_embeddings = self.model.extract_test_embeddings()
        finetuned_eval = self.model.evaluate()
        
        # Guardar resultados post fine-tuning
        self.results['finetuned'] = {
            'embeddings_shape': list(finetuned_embeddings.shape),
            'evaluation': finetuned_eval
        }
        # Guardar embeddings para uso posterior
        self.results['finetuned_embeddings'] = finetuned_embeddings
        
        logging.info(f"Embeddings post fine-tuning extraídos: {finetuned_embeddings.shape}")
        logging.info(f"Accuracy post fine-tuning: {finetuned_eval['accuracy']:.4f}")
        
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
            
            # 6. Calcular mejora
            baseline_acc = self.results['baseline']['evaluation']['accuracy']
            finetuned_acc = self.results['finetuned']['evaluation']['accuracy']
            improvement = finetuned_acc - baseline_acc
            
            self.results['summary'] = {
                'baseline_accuracy': baseline_acc,
                'finetuned_accuracy': finetuned_acc,
                'accuracy_improvement': improvement,
                'improvement_percentage': (improvement / baseline_acc) * 100
            }
            
            logging.info(f"=== PIPELINE COMPLETADO ===")
            logging.info(f"Mejora de accuracy: {improvement:.4f} ({(improvement/baseline_acc)*100:.2f}%)")
            
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
            f"dataset={'✓' if self.dataset else '✗'}, "
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