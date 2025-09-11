import os
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.data.MyDataset import MyDataset
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


class MyVisionModel():
    def __init__(
        self,
        name: str,
        model_name: str,
        weights: str,
        device: torch.device,
        dataset: MyDataset,
        batch_size: int,
    ):
        self.name = name
        self.model_name = model_name
        self.weights = weights

        self.model = models.get_model(self.model_name, weights=self.weights)
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size

        self.train_loader = DataLoader(self.dataset.train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.dataset.val, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.dataset.test, batch_size=self.batch_size)

        self.embedding_dim = None
        self._init_replace_last_layer()
        self.classification_layer = self._init_classification_layer()

    def __str__(self):
        return f"""Model:
                * Name = {self.name}
                * Device = {self.device}
                * Batch size = {self.batch_size}"""

    def _init_replace_last_layer(self)->None:
        """Method for replacing the last layer, for getting the embeddings"""
        if hasattr(self.model, "fc"):  
            self.embedding_dim = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
        elif hasattr(self.model, "heads"):  
            self.embedding_dim = self.model.heads.head.in_features
            self.model.heads = torch.nn.Identity()

    def _init_classification_layer(self) -> torch.nn.Module:
        """Method for setting a new classification layer"""
        head_layer = torch.nn.Linear(self.embedding_dim * self.dataset.num_views, self.dataset.num_models)
        torch.nn.init.xavier_uniform_(head_layer.weight)
        return head_layer

    def extract_embeddings(self, dataloader) -> torch.Tensor:
        """Embeddings extractor of the model"""
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            for batch in tqdm(dataloader, desc='Extracting embeddings', leave=False):
                images = batch['images']
                batch_embeddings = torch.cat([
                    torch.flatten(self.model(image.to(self.device)), start_dim=1)
                    for image in images
                ], dim=1)
                embeddings.append(batch_embeddings.cpu())
            all_embeddings = torch.cat(embeddings, dim=0)
            scaled_embeddings = StandardScaler().fit_transform(all_embeddings.numpy())
        return torch.tensor(scaled_embeddings, dtype=torch.float32)

    def extract_test_embeddings(self) -> torch.Tensor:
        """Extract test embeddings for trials"""
        return self.extract_embeddings(self.test_loader)

    def finetune(
        self,
        Criterion: torch.nn.Module, 
        Optimizer: torch.optim.Optimizer,
        Epochs: int,
        WarmUpEpochs: int = 0
    ) -> None:
        """Fine tuning method"""
        def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
            def lr_lambda(current_epoch):
                return float(current_epoch + 1) / warmup_epochs if current_epoch < warmup_epochs else 1.0
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        self.model.to(self.device)
        self.classification_layer.to(self.device)
        scheduler = get_warmup_scheduler(Optimizer, WarmUpEpochs, Epochs) if WarmUpEpochs else None

        for epoch in range(Epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{Epochs} [Train]', leave=False):
                images = batch['images']
                labels = batch['labels'].to(self.device)

                Optimizer.zero_grad()
                embeddings = torch.cat([
                    torch.flatten(self.model(img.to(self.device)), start_dim=1) for img in images
                ], dim=1)
                outputs = self.classification_layer(embeddings)
                loss = Criterion(outputs, labels)
                loss.backward()
                Optimizer.step()
                train_loss += loss.item()

            if scheduler:
                scheduler.step()

            self.model.eval()
            self.classification_layer.eval()
            correct, total, val_loss = 0, 0, 0.0
            for batch in tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{Epochs} [Val]', leave=False):
                images = batch['images']
                labels = batch['labels'].to(self.device)
                with torch.no_grad():
                    embeddings = torch.cat([
                        torch.flatten(self.model(img.to(self.device)), start_dim=1) for img in images
                    ], dim=1)
                    outputs = self.classification_layer(embeddings)
                    loss = Criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            val_acc = 100 * correct / total
            logging.info(f'Epoch {epoch+1}/{Epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%')