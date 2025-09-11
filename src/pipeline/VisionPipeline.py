import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime
import pickle
from src.config.TransformConfig import TransformConfig
from src.data.MyDataset import MyDataset
from src.models.MyVisionModel import MyVisionModel
from src.utils.DimensionalityReducer import DimensionalityReducer
from src.utils.ClusteringAnalyzer import ClusteringAnalyzer
from src.utils.ClusterVisualizer import ClusterVisualizer
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class VisionPipeline:
    def __init__(self, config_dict, df):
        self.config = config_dict
        self.dataset_df = df
        self.dataset = None
        self.model = None
        self.baseline_embeddings = None
        self.baseline_labels = None
        self.finetune_embeddings = None
        self.finetune_labels = None
        self.results = {}
        
    def load_data(self):
        transforms = TransformConfig(grayscale=self.config.get('gray_scale', False))

        self.dataset = MyDataset(
            df=self.dataset_df,
            views=self.config.get('views', ['front']),
            train_images=self.config.get('train_images', 5),
            val_ratio=self.config.get('val_ratio', 0.5),
            test_ratio=self.config.get('test_ratio', 0.5),
            seed=self.config.get('seed', 3),
            transform=transforms,
            augment=self.config.get('augment', False)
        )

        logging.info(self.dataset) 

    def create_model(self):
        device = torch.device(self.config.get('device', 'cpu'))

        self.model = MyVisionModel(
            name=self.config.get('name', 'Vision Pipeline'),
            model_name=self.config.get('model_name', 'resnet50'),
            weights=self.config.get('weights', 'IMAGENET1K_V1'),
            device=device,
            dataset=self.dataset,
            batch_size=self.config.get('batch_size', 16)
        )

        logging.info(f" === Model {self.config.get('model_name')} created on {device} === ")
        
    def extract_embeddings(self, phase='baseline'):
        if self.model is None:
            raise ValueError("Model not created. Run create_model() first")
        
        test_embeddings = self.model.extract_test_embeddings()
        test_labels = torch.tensor([sample["labels"].item() for sample in self.dataset.test])
        
        if phase == 'baseline':
            self.baseline_embeddings = test_embeddings
            self.baseline_labels = test_labels
            logging.info(f" === Baseline embeddings extracted: {test_embeddings.shape} === ")
        else:
            self.finetune_embeddings = test_embeddings
            self.finetune_labels = test_labels
            logging.info(f" === Post-finetune embeddings extracted: {test_embeddings.shape} === ")
            
        return test_embeddings, test_labels
    
    def run_dimensionality_reduction(self, embeddings=None, labels=None, phase='baseline'):        
        if embeddings is None:
            embeddings = self.baseline_embeddings if phase == 'baseline' else self.finetune_embeddings
        if labels is None:
            labels = self.baseline_labels if phase == 'baseline' else self.finetune_labels
        if embeddings is None:
            raise ValueError(f"No embeddings for {phase}")
            
        reducer = DimensionalityReducer(
            embeddings=embeddings, 
            labels=labels,
            seed=self.config.get('seed', 3),
            optimizer_trials=self.config.get('optimizer_trials', 5),
            available_methods=self.config.get('available reducer methods', ['pca']),
            n_jobs=self.config.get('n_jobs', -1),
            use_incremental=self.config.get('use_incremental', True)
        )

        scores = reducer.compare_methods()
        best_method, best_embeddings = reducer.get_best_result()
        results = {
            'scores': scores,
            'best_method': best_method,
            'best_embeddings': best_embeddings,
            'reducer': reducer
        }
        self.results[f'{phase}_reduction'] = results

        if phase == 'baseline':
            self.baseline_embeddings = best_embeddings
        else:
            self.finetune_embeddings = best_embeddings
            
        logging.info(f" === Dimensionality reduction {phase} - Best method: {best_method} === ")
        return results
    
    def run_clustering(self, embeddings=None, labels=None, phase='baseline'):        
        if embeddings is None:
            embeddings = self.baseline_embeddings if phase == 'baseline' else self.finetune_embeddings
        if labels is None:
            labels = self.baseline_labels if phase == 'baseline' else self.finetune_labels
        if embeddings is None:
            raise ValueError(f"No embeddings for {phase}")
            
        clustering = ClusteringAnalyzer(
            embeddings=embeddings, 
            true_labels=labels, 
            seed=self.config.get('seed', 3),
            optimizer_trials=self.config.get('optimizer_trials', 50),
            available_methods=self.config.get('available clustering methods', ['agglomerative']),
            use_mini_batch=self.config.get('use_mini_batch', True),
            n_jobs=self.config.get('n_jobs', -1),
        )
        
        results = clustering.cluster_all()
        comparison_df = clustering.compare_methods()
        best_method, best_labels = clustering.get_best_result(metric='adjusted_rand_score')

        clustering_results = {
            'results': results,
            'comparison_df': comparison_df,
            'best_method': best_method,
            'best_labels': best_labels,
            'clustering': clustering
        }
        self.results[f'{phase}_clustering'] = clustering_results

        logging.info(f" === Clustering {phase} - Best method: {best_method} === ")
        return clustering_results
    
    def visualize_results(self, phase='baseline', save_path=None):
        embeddings = self.baseline_embeddings if phase == 'baseline' else self.finetune_embeddings
        labels = self.baseline_labels if phase == 'baseline' else self.finetune_labels
        if f'{phase}_clustering' not in self.results:
            raise ValueError(f"No clustering results for {phase}")
            
        clustering_results = self.results[f'{phase}_clustering']

        visualizer = ClusterVisualizer(
            embeddings=embeddings,
            cluster_labels=clustering_results['best_labels'],
            true_labels=labels,
            test_dataset=self.dataset.test,
            label_encoder=self.dataset.label_encoder,
            seed=self.config.get('seed', 3)
        )
        
        visualizer.print_cluster_statistics()
        summary_df = visualizer.get_cluster_summary()
        visualizer.plot_embeddings_overview(reduction_method='tsne')
        visualizer.visualize_good_clusters(n_clusters=self.config.get('clusters to visualize', 3), images_per_cluster=3)
        visualizer.visualize_mixed_clusters(n_clusters=self.config.get('clusters to visualize', 3), max_models_per_cluster=6)
        
        vis_results = {
            'visualizer': visualizer,
            'summary_df': summary_df
        }
        
        self.results[f'{phase}_visualization'] = vis_results
        logging.info(f" === {phase.capitalize()} visualizations generated === ")
        return vis_results
    
    def run_baseline(self):
        logging.info("=== RUNNING BASELINE ===")
        self.load_data()
        self.create_model()
        self.extract_embeddings(phase='baseline')
        self.run_dimensionality_reduction(phase='baseline')
        self.run_clustering(phase='baseline')
        self.visualize_results(phase='baseline')
        logging.info("=== BASELINE COMPLETED ===")
    
    def fine_tune(self):
        if self.model is None:
            raise ValueError("Model not created. Run run_baseline() first")
        
        optimizer_type = self.config.get('finetune optimizer type', 'Adam')
        base_lr = self.config.get('finetune optimizer lr', 1e-4)
        head_lr = self.config.get('finetune optimizer head_lr', 1e-3)
        weight_decay = self.config.get('finetune optimizer weight_decay', 1e-4)  
        
        optimizer_cls = getattr(torch.optim, optimizer_type)
        optimizer_params = [
            {"params": self.model.model.parameters(), "lr": base_lr, "weight_decay": weight_decay}, 
            {"params": self.model.classification_layer.parameters(), "lr": head_lr, "weight_decay": weight_decay}  
        ]
        optimizer = optimizer_cls(optimizer_params)

        criterion_cls = getattr(torch.nn, self.config['finetune criterion'])
        criterion = criterion_cls()

        logging.info(f"=== STARTING FINE-TUNING ({self.config.get('finetune epochs', 10)} epochs) ===")

        self.model.finetune(
            Criterion=criterion,
            Optimizer=optimizer,
            Epochs=self.config.get('finetune epochs', 10),
            WarmUpEpochs=self.config.get('fine tune warm up epochs', 0)
        )

        logging.info(f"=== FINE-TUNING COMPLETED ===")
        self.extract_embeddings(phase='finetune')

    def run_post_finetune(self):
        """Run pipeline after fine-tuning"""
        if self.finetune_embeddings is None:
            raise ValueError("No post-finetune embeddings. Run fine_tune() first")
            
        logging.info("=== RUNNING POST-FINETUNE ===")
        self.run_dimensionality_reduction(phase='finetune')
        self.run_clustering(phase='finetune')
        self.visualize_results(phase='finetune')
        logging.info("=== POST-FINETUNE COMPLETED ===")
    
    def compare_results(self):
        """Baseline vs fine-tuned"""
        if 'baseline_clustering' not in self.results or 'finetune_clustering' not in self.results:
            raise ValueError("Missing baseline or finetune results to compare")
            
        logging.info("\n=== BASELINE vs FINETUNE COMPARISON ===")
        
        baseline_comparison = self.results['baseline_clustering']['comparison_df']
        finetune_comparison = self.results['finetune_clustering']['comparison_df']
        
        logging.info("\nBASELINE:")
        logging.info(baseline_comparison)
        
        logging.info("\nPOST-FINETUNE:")
        logging.info(finetune_comparison)
        
        baseline_best = self.results['baseline_clustering']['best_method']
        finetune_best = self.results['finetune_clustering']['best_method']
        
        logging.info(f"\nBest baseline method: {baseline_best}")
        logging.info(f"Best finetune method: {finetune_best}")
        
        comparison_results = {
            'baseline_comparison': baseline_comparison,
            'finetune_comparison': finetune_comparison,
            'baseline_best': baseline_best,
            'finetune_best': finetune_best
        }
        
        self.results['comparison'] = comparison_results
        return comparison_results
    
    def run_full_pipeline(self, include_finetune=False):
        """Run the full pipeline"""
        self.run_baseline()
        
        if include_finetune:
            self.fine_tune()
            self.run_post_finetune()
            self.compare_results()
    
    def save_results(self, experiment_name, save_dir='results'):
        """Save all automatically"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_path = os.path.join(save_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_path, exist_ok=True)
        
        with open(os.path.join(experiment_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if 'baseline_clustering' in self.results:
            self.results['baseline_clustering']['comparison_df'].to_csv(
                os.path.join(experiment_path, 'baseline_clustering_comparison.csv')
            )
        
        if 'finetune_clustering' in self.results:
            self.results['finetune_clustering']['comparison_df'].to_csv(
                os.path.join(experiment_path, 'finetune_clustering_comparison.csv')
            )
        
        if self.baseline_embeddings is not None:
            torch.save(self.baseline_embeddings, os.path.join(experiment_path, 'baseline_embeddings.pt'))
        if self.finetune_embeddings is not None:
            torch.save(self.finetune_embeddings, os.path.join(experiment_path, 'finetune_embeddings.pt'))
        
        with open(os.path.join(experiment_path, 'results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        
        if self.model is not None:
            model_path = os.path.join(experiment_path, 'model.pth')
            torch.save({
                'state_dict': self.model.model.state_dict(),
                'classification_layer_state_dict': self.model.classification_layer.state_dict()
            }, model_path)
        logging.info(f" === Results saved to: {experiment_path} ===")
        return experiment_path
    
    def load_experiment(self, experiment_path):
        """Load saved experiment"""
        with open(os.path.join(experiment_path, 'config.json'), 'r') as f:
            self.config = json.load(f)
            
        with open(os.path.join(experiment_path, 'results.pkl'), 'rb') as f:
            self.results = pickle.load(f)
            
        baseline_path = os.path.join(experiment_path, 'baseline_embeddings.pt')
        if os.path.exists(baseline_path):
            self.baseline_embeddings = torch.load(baseline_path)
            
        finetune_path = os.path.join(experiment_path, 'finetune_embeddings.pt')
        if os.path.exists(finetune_path):
            self.finetune_embeddings = torch.load(finetune_path)
            
        logging.info(f" === Experiment loaded from: {experiment_path} === ")
    
    def compare_experiments(self, experiment_paths, metric='adjusted_rand_score'):
        """Compare multiple experiments"""
        comparison_data = []
        
        for exp_path in experiment_paths:
            exp_name = os.path.basename(exp_path)
            
            baseline_path = os.path.join(exp_path, 'baseline_clustering_comparison.csv')
            if os.path.exists(baseline_path):
                baseline_df = pd.read_csv(baseline_path)
                best_baseline = baseline_df.loc[baseline_df[metric].idxmax()]
                
                comparison_data.append({
                    'experiment': exp_name,
                    'phase': 'baseline',
                    'best_method': best_baseline['method'],
                    metric: best_baseline[metric]
                })
            
            finetune_path = os.path.join(exp_path, 'finetune_clustering_comparison.csv')
            if os.path.exists(finetune_path):
                finetune_df = pd.read_csv(finetune_path)
                best_finetune = finetune_df.loc[finetune_df[metric].idxmax()]
                
                comparison_data.append({
                    'experiment': exp_name,
                    'phase': 'finetune',
                    'best_method': best_finetune['method'],
                    metric: best_finetune[metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        logging.info("\n=== CROSS-EXPERIMENT COMPARISON ===")
        logging.info(comparison_df)
        return comparison_df