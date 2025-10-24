"""
Script de Fine-Tuning para ResNet50.

Este script ejecuta el proceso completo de fine-tuning para el modelo ResNet50
usando configuración desde archivo YAML con logging detallado.
"""

import pandas as pd
import yaml
import logging
import sys

from src.pipeline.FineTuningPipeline import create_finetuning_pipeline


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("="*80)
    logging.info("INICIANDO SCRIPT DE FINE-TUNING RESNET50")
    logging.info("="*80)

    try:
        # Rutas de archivos
        dataset_csv_path = "dataset.csv" 
        config_yaml_path = "configs/resnet50_classification.yaml"
        
        # Cargar dataset
        logging.info(f"\n{'='*80}")
        logging.info("CARGANDO DATASET")
        logging.info(f"{'='*80}")
        logging.info(f"Ruta: {dataset_csv_path}")
        
        dataset_df = pd.read_csv(dataset_csv_path)
        logging.info(f"Dataset cargado")
        
        # Cargar configuración
        logging.info(f"\n{'='*80}")
        logging.info("CARGANDO CONFIGURACIÓN")
        logging.info(f"{'='*80}")
        
        logging.info(f"Cargando configuración")
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("Configuración cargada exitosamente")

        # Crear pipeline de fine-tuning
        logging.info(f"\n{'='*80}")
        logging.info("CREANDO PIPELINE")
        logging.info(f"{'='*80}")
        
        pipeline = create_finetuning_pipeline(
            config=config,
            df=dataset_df,
            experiment_name="resnet50_classification",
            model_type='vision'
        )
        
        logging.info("Pipeline creado exitosamente")
        
        results = pipeline.run_full_pipeline()
        
        logging.info(f"\n{'='*80}")
        logging.info("SCRIPT FINALIZADO CORRECTAMENTE")
        logging.info(f"{'='*80}")
        
    except FileNotFoundError as e:
        logging.error(f"\n{'='*80}")
        logging.error("ERROR: ARCHIVO NO ENCONTRADO")
        logging.error(f"{'='*80}")
        logging.error(f"Detalle: {e}")
        logging.error("Verifique que los archivos existan en las rutas especificadas")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"\n{'='*80}")
        logging.error("ERROR DURANTE LA EJECUCIÓN")
        logging.error(f"{'='*80}")
        logging.error(f"Tipo: {type(e).__name__}")
        logging.error(f"Mensaje: {e}")
        logging.exception("Stack trace completo:")
        sys.exit(1)