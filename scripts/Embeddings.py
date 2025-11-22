"""
Script de estudio de embeddings para modelos de visión.

Este script ejecuta el análisis de embeddings sobre todos los experimentos de fine-tuning
guardados en el directorio de resultados, aplicando la configuración universal de embeddings.
"""

import json
import pandas as pd
import yaml
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

from src.pipeline.EmbeddingsPipeline import create_embeddings_pipeline

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("="*80)
    logging.info("INICIANDO SCRIPT DE ANÁLISIS DE EMBEDDINGS")
    logging.info("="*80)

    try:
        # Rutas de archivos
        dataset_csv_path = "dataset.csv"
        embeddings_config_path = "configs/embeddings.yaml"
        results_dir = Path("results")
        
        # Cargar dataset
        logging.info(f"\n{'='*80}")
        logging.info("CARGANDO DATASET")
        logging.info(f"{'='*80}")
        logging.info(f"Ruta: {dataset_csv_path}")
        
        dataset_df = pd.read_csv(dataset_csv_path)
        logging.info(f"Dataset cargado: {len(dataset_df)} registros")
        
        # Cargar configuración de embeddings
        logging.info(f"\n{'='*80}")
        logging.info("CARGANDO CONFIGURACIÓN DE EMBEDDINGS")
        logging.info(f"{'='*80}")
        logging.info(f"Ruta: {embeddings_config_path}")
        
        with open(embeddings_config_path, 'r', encoding='utf-8') as f:
            embeddings_config = yaml.safe_load(f)
        logging.info("Configuración de embeddings cargada")
        
        # Obtener todos los archivos ZIP del directorio de resultados
        zip_files = sorted(results_dir.glob("finetune_*.zip"))
        
        if not zip_files:
            logging.warning(f"No se encontraron archivos ZIP en {results_dir}")
            sys.exit(0)
        
        logging.info(f"\n{'='*80}")
        logging.info(f"ARCHIVOS ZIP ENCONTRADOS: {len(zip_files)}")
        logging.info(f"{'='*80}")
        for zip_file in zip_files:
            logging.info(f"  - {zip_file.name}")
        
        # *************** BORRAR AQUI: Procesar solo el primer ZIP para pruebas ***************
        logging.info(f"\n{'='*80}")
        logging.info("MODO DE PRUEBA: PROCESANDO SOLO EL PRIMER ZIP")
        logging.info(f"{'='*80}")
        zip_files = [zip_files[0]]
        # *************** BORRAR AQUI ***************
        
        # Procesar cada archivo ZIP
        successful_analyses = 0
        failed_analyses = []
        
        for idx, zip_path in enumerate(zip_files, 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"PROCESANDO ZIP {idx}/{len(zip_files)}: {zip_path.name}")
            logging.info(f"{'='*80}")
            
            try:
                # Cargar configuración del ZIP
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_path)
                    
                    config_files = list(temp_path.rglob("config.json"))
                    
                    if not config_files:
                        raise FileNotFoundError(f"No se encontró config.json en {zip_path.name}")
                    
                    with open(config_files[0], 'r') as f:
                        zip_config = json.load(f)
                    
                    logging.info("Configuración del ZIP cargada")
                
                # Fusionar configuraciones: embeddings_config sobreescribe cualquier parámetro del ZIP
                merged_config = {**zip_config, **embeddings_config}
                
                logging.info("Configuraciones fusionadas (embeddings_config tiene prioridad)")
                
                # Crear pipeline con la configuración fusionada
                logging.info(f"\n{'='*80}")
                logging.info("CREANDO PIPELINE DE EMBEDDINGS")
                logging.info(f"{'='*80}")
                
                pipeline = create_embeddings_pipeline(
                    config=merged_config,
                    df=dataset_df
                )
                
                logging.info("Pipeline creado exitosamente")
                
                # Ejecutar análisis completo desde el ZIP
                results = pipeline.run_full_analysis_from_zip(zip_path=zip_path)
                
                successful_analyses += 1
                
                logging.info(f"\n{'='*80}")
                logging.info(f"ANÁLISIS COMPLETADO: {zip_path.name}")
                logging.info(f"Resultados guardados en: {results.get('saved_to', 'N/A')}")
                logging.info(f"{'='*80}")
                
            except Exception as e:
                failed_analyses.append((zip_path.name, str(e)))
                logging.error(f"\n{'='*80}")
                logging.error(f"ERROR PROCESANDO: {zip_path.name}")
                logging.error(f"{'='*80}")
                logging.error(f"Tipo: {type(e).__name__}")
                logging.error(f"Mensaje: {e}")
                logging.exception("Stack trace completo:")
                continue
        
        # Resumen final
        logging.info(f"\n{'='*80}")
        logging.info("RESUMEN DE EJECUCIÓN")
        logging.info(f"{'='*80}")
        logging.info(f"Total de ZIPs procesados: {len(zip_files)}")
        logging.info(f"Análisis exitosos: {successful_analyses}")
        logging.info(f"Análisis fallidos: {len(failed_analyses)}")
        
        if failed_analyses:
            logging.warning("\nArchivos con errores:")
            for zip_name, error in failed_analyses:
                logging.warning(f"  - {zip_name}: {error}")
        
        logging.info(f"\n{'='*80}")
        logging.info("SCRIPT FINALIZADO")
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
