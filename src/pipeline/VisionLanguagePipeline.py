import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os
from datetime import datetime
import pickle
from src.config.TransformConfig import TransformConfig
from src.data.MyDataset import MyDataset
from src.models.MyVisionLanguageModel import MyVisionLanguageModel
from src.pipeline.VisionPipeline import VisionPipeline
import warnings
import logging

import warnings
warnings.filterwarnings('ignore')

import logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisionLanguagePipeline(VisionPipeline):
    def __init__(self, config_dict, df):
        super().__init__(config_dict, df)

    def load_data(self):
        """Override"""
        transforms = TransformConfig(grayscale=self.config.get('gray_scale', False))

        self.dataset = MyDataset(
            df=self.dataset_df,
            views=self.config.get('views', ['front']),
            train_images=self.config.get('train_images', 5),
            val_ratio=self.config.get('val_ratio', 0.5),
            test_ratio=self.config.get('test_ratio', 0.5),
            seed=self.config.get('seed', 3),
            transform=transforms,
            augment=self.config.get('augment', False),
            model_type=self.config.get('model_type', 'both'),
            description_include=self.config.get('description_include', 'all')
        )
        logging.info(self.dataset)

    def create_model(self):
        """Override"""
        device = torch.device(self.config.get('device', 'cpu'))

        self.model = MyVisionLanguageModel(
            name=self.config.get('name', 'Vision-Language Pipeline'),
            model_name=self.config.get('model_name', 'openai/clip-vit-base-patch16'),
            device=device,
            dataset=self.dataset,
            batch_size=self.config.get('batch_size', 8),
            vision_weights_path=self.config.get('vision_weights_path', None)
        )
        logging.info(f" === Vision-Language Model {self.config.get('model_name')} created on {device} === ")

    def extract_embeddings(self, phase='baseline', embedding_type='combined'):
        """
        Extract embeddings from the vision-language model
        
        Args:
            phase: 'baseline' or 'finetune'
            embedding_type: 'vision', 'text', or 'combined'
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if embedding_type == 'vision':
            test_embeddings = self.extract_vision_embeddings()
        elif embedding_type == 'text':
            test_embeddings = self.extract_text_embeddings()
        elif embedding_type == 'combined':
            test_embeddings = self.extract_combined_embeddings()
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}. Use 'vision', 'text', or 'combined'")
        
        test_labels = torch.tensor([sample["labels"].item() for sample in self.dataset.test])
        
        if phase == 'baseline':
            self.baseline_embeddings = test_embeddings
            self.baseline_labels = test_labels
            logging.info(f"Baseline {embedding_type} embeddings extracted: {test_embeddings.shape}")
        else:
            self.finetune_embeddings = test_embeddings
            self.finetune_labels = test_labels
            logging.info(f"Fine-tuned {embedding_type} embeddings extracted: {test_embeddings.shape}")
        
    def extract_vision_embeddings(self):
        """Extract vision-only embeddings (using vision adapter)"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.model.extract_vision_embeddings(self.model.test_loader)

    def extract_text_embeddings(self):
        """Extract text-only embeddings"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.model.extract_text_embeddings(self.model.test_loader)

    def extract_combined_embeddings(self):
        """Extract combined vision+text embeddings"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.model.extract_embeddings(self.model.test_loader)

    def run_baseline(self, embedding_type='combined'):
        logging.info(f"=== RUNNING BASELINE ({embedding_type.upper()} EMBEDDINGS) ===")
        self.load_data()
        self.create_model()
        self.extract_embeddings(phase='baseline', embedding_type=embedding_type)
        self.run_dimensionality_reduction(phase='baseline')
        self.run_clustering(phase='baseline')
        self.visualize_results(phase='baseline')
        logging.info(f"=== BASELINE COMPLETED ({embedding_type.upper()}) ===")

    def fine_tune(self):
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        optimizer_type = self.config.get('finetune optimizer type', 'AdamW')
        base_lr = self.config.get('finetune optimizer lr', 5e-6)
        head_lr = self.config.get('finetune optimizer head_lr', 5e-6)
        weight_decay = self.config.get('finetune optimizer weight_decay', 1e-2)  
        betas = self.config.get('betas', (0.9, 0.98))

        optimizer_cls = getattr(torch.optim, optimizer_type)
        optimizer_params = [
            {"params": self.model.model.text_model.parameters(), "lr": base_lr, "weight_decay": weight_decay, "betas": betas}, 
            {"params": self.model.text_classification_layer.parameters(), "lr": head_lr, "weight_decay": weight_decay, "betas":betas}  
        ]

        optimizer = optimizer_cls(optimizer_params)

        logging.info(f"=== STARTING FINE-TUNING ({self.config.get('finetune epochs', 10)} epochs) ===")

        self.model.finetune(
            Optimizer=optimizer,
            Epochs=self.config.get('finetune epochs', 10),
            WarmUpEpochs=self.config.get('fine tune warm up epochs', 0),
            Temperature=self.config.get('temperature', 0.07)
        )
        logging.info(f"=== FINE-TUNING COMPLETED ===")
        self.extract_embeddings(phase='finetune')
        
    def run_post_finetune(self, embedding_type='vision'):
        """
        Run analysis after fine-tuning
        
        Args:
            embedding_type: 'vision', 'text', or 'combined' - which embeddings to use
        """
        if self.finetune_embeddings is None:
            logging.warning("No fine-tuned embeddings found. Running fine-tuning first...")
            self.fine_tune()
            
        logging.info(f"=== RUNNING POST-FINETUNE ({embedding_type.upper()} EMBEDDINGS) ===")
        self.extract_embeddings(phase='finetune', embedding_type=embedding_type)
        self.run_dimensionality_reduction(phase='finetune')
        self.run_clustering(phase='finetune')
        self.visualize_results(phase='finetune')
        logging.info("=== POST-FINETUNE COMPLETED ===")

    def save_results(self, experiment_name, save_dir='results'):
        """Save all results including vision and text weights separately"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_path = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_path, exist_ok=True)
        
        with open(os.path.join(experiment_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if 'baseline_clustering' in self.results:
            self.results['baseline_clustering']['comparison_df'].to_csv(
                os.path.join(experiment_path, 'baseline_clustering_results.csv'), index=False
            )
        
        if 'finetune_clustering' in self.results:
            self.results['finetune_clustering']['comparison_df'].to_csv(
                os.path.join(experiment_path, 'finetune_clustering_results.csv'), index=False
            )
        
        if self.baseline_embeddings is not None:
            torch.save(self.baseline_embeddings, os.path.join(experiment_path, 'baseline_embeddings.pt'))
        if self.finetune_embeddings is not None:
            torch.save(self.finetune_embeddings, os.path.join(experiment_path, 'finetune_embeddings.pt'))
        
        with open(os.path.join(experiment_path, 'results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        self.model.save_weights(save_dir)
        logging.info(f" === Vision-Language results saved to: {experiment_path} ===")
        return experiment_path