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

warnings.filterwarnings('ignore')

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VisionLanguagePipeline(VisionPipeline):
    def __init__(self, config_dict, df):
        super().__init__(config_dict, df)

    def load_data(self):
        """Override parent method to load data for vision-language tasks"""
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
        """Override parent method to create vision-language model"""
        device = torch.device(self.config.get('device', 'cpu'))
        self.model = MyVisionLanguageModel(
            name=self.config.get('name', 'Vision-Language Pipeline'),
            clip_model_name=self.config.get('clip_model_name', 'openai/clip-vit-base-patch16'),
            device=device,
            dataset=self.dataset,
            batch_size=self.config.get('batch_size', 8),
            vision_model_name=self.config.get('vision_model_name', 'resnet50'),
            vision_weights_path=self.config.get('vision_weights_path', None)
        )
        logging.info(f" === Vision-Language Model {self.config.get('clip_model_name', 'openai/clip-vit-base-patch16')} created on {device} === ")

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
            test_embeddings = self.model.extract_vision_embeddings(self.model.test_loader)
        elif embedding_type == 'text':
            test_embeddings = self.model.extract_text_embeddings(self.model.test_loader)
        elif embedding_type == 'combined':
            test_embeddings = self.model.extract_embeddings(self.model.test_loader)
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

    def run_dimensionality_reduction(self, phase='baseline', embeddings=None, labels=None):
        """Override parent method to ensure consistency"""
        if embeddings is None:
            embeddings = self.baseline_embeddings if phase == 'baseline' else self.finetune_embeddings
        if labels is None:
            labels = self.baseline_labels if phase == 'baseline' else self.finetune_labels
        if embeddings is None:
            raise ValueError(f"No embeddings for {phase}")
            
        logging.info(f"Running dimensionality reduction on embeddings: {embeddings.shape}")
        results = super().run_dimensionality_reduction(embeddings=embeddings, labels=labels, phase=phase)
        self.results[f'{phase}_dimensionality_reduction'] = results.copy()
        self.results[f'{phase}_dimensionality_reduction']['embeddings'] = results['best_embeddings']
        
        return results

    def run_clustering(self, phase='baseline', embeddings=None, labels=None):
        """Override parent method with better error handling"""
        if embeddings is None:
            if f'{phase}_reduction' in self.results:
                embeddings = self.results[f'{phase}_reduction']['best_embeddings']
                logging.info(f"Using reduced embeddings for clustering: {embeddings.shape}")
            else:
                if phase == 'baseline':
                    embeddings = self.baseline_embeddings
                elif phase == 'finetune':
                    embeddings = self.finetune_embeddings
                else:
                    raise ValueError(f"Invalid phase: {phase}. Use 'baseline' or 'finetune'")
                logging.info(f"Using original embeddings for clustering: {embeddings.shape}")
        
        if labels is None:
            labels = self.baseline_labels if phase == 'baseline' else self.finetune_labels
        
        if embeddings is None or labels is None:
            raise ValueError(f"No embeddings or labels found for phase: {phase}")
        
        logging.info(f"Clustering with embeddings: {embeddings.shape}, labels: {labels.shape}")
        super().run_clustering(embeddings=embeddings, labels=labels, phase=phase)

    def run_baseline(self, embedding_type='combined'):
        """Run baseline analysis with specified embedding type"""
        logging.info(f"=== RUNNING BASELINE ({embedding_type.upper()} EMBEDDINGS) ===")
        self.load_data()
        self.create_model()
        self.extract_embeddings(phase='baseline', embedding_type=embedding_type)
        self.run_dimensionality_reduction(phase='baseline')
        if hasattr(self, 'results') and 'baseline_dimensionality_reduction' in self.results:
            reduced_embeddings = self.results['baseline_dimensionality_reduction']['embeddings']        
        self.run_clustering(phase='baseline')
        self.visualize_results(phase='baseline')
        logging.info(f"=== BASELINE COMPLETED ({embedding_type.upper()}) ===")

    def fine_tune(self):
        """Fine-tune the vision-language model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        optimizer_type = self.config.get('finetune_optimizer_type', 'AdamW')
        base_lr = self.config.get('finetune_optimizer_lr', 5e-6)
        head_lr = self.config.get('finetune_optimizer_head_lr', 5e-5)
        weight_decay = self.config.get('finetune_optimizer_weight_decay', 1e-2)  
        betas = self.config.get('finetune_optimizer_betas', (0.9, 0.98))

        optimizer_cls = getattr(torch.optim, optimizer_type)
        optimizer_params = [
            {
                "params": self.model.model.text_model.parameters(), 
                "lr": base_lr, 
                "weight_decay": weight_decay
            },
            {
                "params": self.model.model.text_projection.parameters(), 
                "lr": base_lr, 
                "weight_decay": weight_decay
            },
            {
                "params": self.model.text_classification_layer.parameters(), 
                "lr": head_lr, 
                "weight_decay": weight_decay
            }
        ]
        
        if optimizer_type == 'AdamW':
            for param_group in optimizer_params:
                param_group['betas'] = betas

        optimizer = optimizer_cls(optimizer_params)

        criterion_cls = getattr(torch.nn, self.config.get('finetune_criterion', 'CrossEntropyLoss'))
        criterion = criterion_cls()

        logging.info(f"=== STARTING FINE-TUNING ({self.config.get('finetune_epochs', 10)} epochs) ===")

        self.model.finetune(
            Criterion=criterion,
            Optimizer=optimizer,
            Epochs=self.config.get('finetune_epochs', 10),
            WarmUpSteps=self.config.get('finetune_warm_up_steps', 100),
            Temperature=self.config.get('finetune_temperature', 0.07)
        )
        logging.info(f"=== FINE-TUNING COMPLETED ===")
        
    def run_post_finetune(self, embedding_type='combined'):
        """
        Run analysis after fine-tuning
        
        Args:
            embedding_type: 'vision', 'text', or 'combined' - which embeddings to use
        """
        logging.info(f"=== RUNNING POST-FINETUNE ({embedding_type.upper()} EMBEDDINGS) ===")
        self.extract_embeddings(phase='finetune', embedding_type=embedding_type)
        self.run_dimensionality_reduction(phase='finetune')
        self.run_clustering(phase='finetune')
        self.visualize_results(phase='finetune')
        logging.info("=== POST-FINETUNE COMPLETED ===")

    def run_full_pipeline(self, embedding_type='combined'):
        """
        Run complete pipeline: baseline -> fine-tune -> post-finetune analysis
        
        Args:
            embedding_type: 'vision', 'text', or 'combined'
        """
        logging.info(f"=== STARTING FULL PIPELINE ({embedding_type.upper()}) ===")
        self.run_baseline(embedding_type=embedding_type)
        self.fine_tune()
        self.run_post_finetune(embedding_type=embedding_type)
        logging.info(f"=== FULL PIPELINE COMPLETED ({embedding_type.upper()}) ===")

    def save_results(self, experiment_name, save_dir='results'):
        """Save all results including model weights"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_path = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_path, exist_ok=True)
        
        with open(os.path.join(experiment_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if hasattr(self, 'results'):
            if 'baseline_clustering' in self.results:
                self.results['baseline_clustering']['comparison_df'].to_csv(
                    os.path.join(experiment_path, 'baseline_clustering_results.csv'), index=False
                )
            
            if 'finetune_clustering' in self.results:
                self.results['finetune_clustering']['comparison_df'].to_csv(
                    os.path.join(experiment_path, 'finetune_clustering_results.csv'), index=False
                )
        
        if hasattr(self, 'baseline_embeddings') and self.baseline_embeddings is not None:
            torch.save(self.baseline_embeddings, os.path.join(experiment_path, 'baseline_embeddings.pt'))
        if hasattr(self, 'finetune_embeddings') and self.finetune_embeddings is not None:
            torch.save(self.finetune_embeddings, os.path.join(experiment_path, 'finetune_embeddings.pt'))
        
        if hasattr(self, 'results'):
            with open(os.path.join(experiment_path, 'results.pkl'), 'wb') as f:
                pickle.dump(self.results, f)
        
        if self.model is not None:
            model_weights_path = os.path.join(experiment_path, 'model_weights')
            self.model.save_weights(model_weights_path)
        
        logging.info(f" === Vision-Language results saved to: {experiment_path} ===")
        return experiment_path

    def load_results(self, experiment_path):
        """Load previously saved results"""
        config_path = os.path.join(experiment_path, 'config.json')
        results_path = os.path.join(experiment_path, 'results.pkl')
        baseline_embeddings_path = os.path.join(experiment_path, 'baseline_embeddings.pt')
        finetune_embeddings_path = os.path.join(experiment_path, 'finetune_embeddings.pt')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                logging.info("Configuration loaded successfully")
        
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                self.results = pickle.load(f)
                logging.info("Results loaded successfully")
        
        if os.path.exists(baseline_embeddings_path):
            self.baseline_embeddings = torch.load(baseline_embeddings_path)
            logging.info(f"Baseline embeddings loaded: {self.baseline_embeddings.shape}")
        
        if os.path.exists(finetune_embeddings_path):
            self.finetune_embeddings = torch.load(finetune_embeddings_path)
            logging.info(f"Fine-tuned embeddings loaded: {self.finetune_embeddings.shape}")
        
        model_weights_path = os.path.join(experiment_path, 'model_weights')
        if os.path.exists(model_weights_path) and self.model is not None:
            self.model.load_weights(model_weights_path)
            logging.info("Model weights loaded successfully")
        
        return loaded_config if os.path.exists(config_path) else None