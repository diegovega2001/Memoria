import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.MyDataset import MyDataset
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup
import warnings
import logging
from src.models.MyVisionModel import MyVisionModel

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)



class MyVisionLanguageModel(torch.nn.Module):
    def __init__(
        self,
        name: str,
        clip_model_name: str,
        device: torch.device,
        dataset: MyDataset,
        batch_size: int,
        vision_model_name: str,
        vision_weights_path: str
    ):
        super().__init__()
        self.name = name
        self.clip_model_name = clip_model_name
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.vision_model_name = vision_model_name
        self.vision_weights_path = vision_weights_path

        logging.info(f"Loading CLIP model: {clip_model_name}")
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)

        self.original_vision_dim = self.model.vision_model.config.hidden_size
        
        self.model.vision_model = MyVisionModel(
            name="vision_encoder",
            model_name=self.vision_model_name,
            weights="IMAGENET1K_V1",  
            device=device,
            dataset=dataset,
            batch_size=batch_size
        )
        
        self.vision_adapter = nn.Linear(self.model.vision_model.embedding_dim * self.dataset.num_views, self.original_vision_dim)
        
        self.model.to(self.device)
        self.vision_adapter.to(self.device)
        
        def custom_collate_fn(batch):
            images = [sample['images'] for sample in batch]  
            labels = torch.stack([sample['labels'] for sample in batch])
            texts = [sample['text_description'] for sample in batch]
            return {'images': images, 'labels': labels, 'text_description': texts}
        
        self.train_loader = DataLoader(self.dataset.train, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(self.dataset.val, batch_size=self.batch_size, collate_fn=custom_collate_fn)
        self.test_loader = DataLoader(self.dataset.test, batch_size=self.batch_size, collate_fn=custom_collate_fn)
        
        self.vision_embedding_dim = self.original_vision_dim  
        self.text_embedding_dim = self.model.text_model.config.hidden_size
        self.projection_dim = self.model.projection_dim
            
        self.load_vision_model_weights(self.vision_weights_path)
        self._freeze_vision_model()
        self.text_classification_layer = self._init_text_classification_layer()

    def _get_vision_features(self, images) -> torch.Tensor:
        """Extract and adapt vision features from MyVisionModel for CLIP compatibility"""
        raw_features = self.model.vision_model(images)
        if raw_features.ndim > 2:
            raw_features = raw_features.view(raw_features.size(0), -1)

        adapted_features = self.vision_adapter(raw_features)
        return adapted_features
    
    def _init_text_classification_layer(self) -> torch.nn.Module:
        """Initialize classification layer for text embeddings (similar to MyVisionModel)"""
        head_layer = torch.nn.Linear(self.text_embedding_dim, self.dataset.num_models)
        torch.nn.init.xavier_uniform_(head_layer.weight)
        return head_layer

    def load_vision_model_weights(self, weights_path=None):
        if weights_path:
            self.model.vision_model.load_weights(weights_path)
    
    def __str__(self):
        return f"""Vision-Language Model:
                * Name = {self.name}
                * Model = {self.model_name}
                * Device = {self.device}
                * Batch size = {self.batch_size}
                * Vision embedding dim = {self.vision_embedding_dim}
                * Text embedding dim = {self.text_embedding_dim}
                * Projection dim = {self.projection_dim}
                * MyVisionModel embedding_dim = {self.my_vision_model.embedding_dim}
                * Adaptation layer = {self.my_vision_model.embedding_dim * self.dataset.num_views} -> {self.original_vision_dim}"""

    def _freeze_vision_model(self) -> None:
        """Freeze the vision model parameters (similar to MyVisionModel pattern)"""
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.vision_adapter.parameters():
            param.requires_grad = False
        logging.info("MyVisionModel and adapter frozen for text-only fine-tuning")

    def _unfreeze_text_model(self) -> None:
        """Unfreeze text model parameters for fine-tuning"""
        for param in self.model.text_model.parameters():
            param.requires_grad = True
        logging.info("Text model unfrozen for fine-tuning")

    def extract_vision_embeddings(self, dataloader) -> torch.Tensor:
        """Extract visual embeddings and project them to CLIP latent space."""
        self.model.vision_model.to(self.device)
        self.model.vision_model.eval()
        self.vision_adapter.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting vision embeddings', leave=False):
                images = batch['images']
                vision_features = self._get_vision_features(images)
                vision_embeds = self.model.visual_projection(vision_features)
                embeddings.append(vision_embeds.cpu())
        all_embeddings = torch.cat(embeddings, dim=0)
        all_embeddings /= all_embeddings.norm(dim=1, keepdim=True) 
        return torch.tensor(all_embeddings, dtype=torch.float32)
                                                          
    def extract_text_embeddings(self, dataloader) -> torch.Tensor:
        """Extract text embeddings and proyect them to CLIP latent space."""
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting text embeddings', leave=False):
                texts = batch['text_description']
                processed_texts = self.processor(
                    text=texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=77
                )
                input_ids = processed_texts['input_ids'].to(self.device)
                attention_mask = processed_texts['attention_mask'].to(self.device)
                text_outputs = self.model.text_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                text_embeds = self.model.text_projection(text_outputs.pooler_output)
                embeddings.append(text_embeds.cpu())
        all_embeddings = torch.cat(embeddings, dim=0)
        all_embeddings /= all_embeddings.norm(dim=1, keepdim=True) 
        return torch.tensor(all_embeddings, dtype=torch.float32)

    def extract_embeddings(self, dataloader) -> torch.Tensor:
        """Extract combined vision and text embeddings"""
        vision_embeddings = self.extract_vision_embeddings(dataloader)
        text_embeddings = self.extract_text_embeddings(dataloader)
        
        combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
        return combined_embeddings
    
    def extract_test_embeddings(self) -> torch.Tensor:
        """Extract test embeddings (following MyVisionModel pattern)"""
        return self.extract_embeddings(self.test_loader)

    def finetune(
        self,
        Criterion: torch.nn.Module,
        Optimizer: torch.optim.Optimizer,
        Epochs: int,
        WarmUpSteps: int = 0,
        Temperature: float = 0.07
    ) -> None:
        """Fine-tune CLIP text encoder."""
        total_steps = Epochs * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(
            Optimizer, 
            num_warmup_steps=WarmUpSteps, 
            num_training_steps=total_steps
        ) if WarmUpSteps > 0 else None

        self.model.to(self.device)
        self.text_classification_layer.to(self.device)
        self._freeze_vision_model()
        self._unfreeze_text_model()

        for epoch in range(Epochs):
            self.model.train()
            self.text_classification_layer.train()
            train_steps = 0
            train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{Epochs} [Train]', leave=False):
                images = batch['images'] 
                texts = batch['text_description']
                labels = batch['labels'].to(self.device)

                with torch.no_grad():
                    vision_features = self._get_vision_features(images)
                    image_embeds = self.model.visual_projection(vision_features)
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                processed = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )
                input_ids = processed['input_ids'].to(self.device)
                attention_mask = processed['attention_mask'].to(self.device)
                text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeds = self.model.text_projection(text_outputs.pooler_output)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                logits_per_image = (image_embeds @ text_embeds.t()) / Temperature
                logits_per_text = logits_per_image.t()
                contrastive_targets = torch.arange(len(images)).to(self.device)

                loss_i = nn.CrossEntropyLoss()(logits_per_image, contrastive_targets)
                loss_t = nn.CrossEntropyLoss()(logits_per_text, contrastive_targets)
                contrastive_loss = (loss_i + loss_t) / 2

                cls_loss = 0.0
                cls_outputs = self.text_classification_layer(text_embeds)
                cls_loss = Criterion(cls_outputs, labels)

                loss = contrastive_loss + cls_loss

                Optimizer.zero_grad()
                loss.backward()
                Optimizer.step()
                if scheduler:
                    scheduler.step()

                train_loss += loss.item()
                train_steps += 1

            self.model.eval()
            self.text_classification_layer.eval()
            val_steps = 0
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{Epochs} [Val]', leave=False):
                    images = batch['images']
                    texts = batch['text_description']
                    labels = batch['labels'].to(self.device)
                    
                    vision_features = self._get_vision_features(images)
                    image_embeds = self.model.visual_projection(vision_features)
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

                    processed = self.processor(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    )
                    input_ids = processed['input_ids'].to(self.device)
                    attention_mask = processed['attention_mask'].to(self.device)
                    text_outputs = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                    text_embeds = self.model.text_projection(text_outputs.pooler_output)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                    logits_per_image = (image_embeds @ text_embeds.t()) / Temperature
                    logits_per_text = logits_per_image.t()
                    contrastive_targets = torch.arange(len(images)).to(self.device)

                    loss_i = nn.CrossEntropyLoss()(logits_per_image, contrastive_targets)
                    loss_t = nn.CrossEntropyLoss()(logits_per_text, contrastive_targets)
                    contrastive_loss = (loss_i + loss_t) / 2

                    cls_outputs = self.text_classification_layer(text_embeds)
                    cls_loss = Criterion(cls_outputs, labels)
                    
                    _, predicted = cls_outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = contrastive_loss + cls_loss
                    val_loss += loss.item()
                    val_steps += 1

            avg_train_loss = train_loss / max(1, train_steps)
            avg_val_loss = val_loss / max(1, val_steps)
            val_acc = 100.0 * correct / total if total > 0 else 0.0
            logging.info(f"Epoch {epoch+1}/{Epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc (text cls): {val_acc:.2f}%")

    def save_weights(self, weights_path):
        os.makedirs(weights_path, exist_ok=True)
        model_path = os.path.join(weights_path, 'text_model.pth')
        torch.save({
            'text_model_state_dict': self.model.text_model.state_dict(),
            'text_projection_state_dict': self.model.text_projection.state_dict(),
            'text_classification_layer_state_dict': self.text_classification_layer.state_dict()
        }, model_path)
        logging.info(f"Text model weights saved to: {model_path}")
    
    def load_weights(self, weights_path):
        model_path = os.path.join(weights_path, 'text_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.text_model.load_state_dict(checkpoint['text_model_state_dict'])
            self.model.text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
            self.text_classification_layer.load_state_dict(checkpoint['text_classification_layer_state_dict'])
            logging.info(f"Text model weights loaded from: {model_path}")
        else:
            logging.warning(f"No weights found at: {model_path}")
            logging.info("Starting with pre-trained CLIP text model weights")
