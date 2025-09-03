import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torchvision.transforms.functional as F
import torch


class MyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        views: list[str],
        train_images: int = 0,
        val_ratio: float = 0.0,
        test_ratio: float = 0.0,
        seed: int = 3,
        transform = None,
        augment: bool = False,
    ):
        self.df = df.copy()
        self.views = views
        self.num_views = len(self.views)
        self.train_images = train_images
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_images = train_images
        self.seed = seed
        self.transform = transform
        self.augment = augment

        self.models = self._init_models()
        self.num_models = len(self.models)

        self.label_encoder = self._init_label_encoder()

        self.df = self._update_df()
        
        self.train, self.val, self.test = self._init_split_()

    def _init_models(self):
        counts = (
            self.df.groupby(["model", "viewpoint"])
            .size()
            .unstack(fill_value=0)
        )
        models = counts[
            (counts[self.views] >= self.min_images).all(axis=1)
        ].index
        return models
    
    def _init_label_encoder(self):
        label_encoder = LabelEncoder()
        label_encoder.fit(self.models)
        return label_encoder
    
    def _update_df(self):
        updated_df = self.df.copy()
        updated_df = updated_df[
            (updated_df["model"].isin(self.models))
            & (updated_df["viewpoint"].isin(self.views))
        ]
        return updated_df

    def _init_split_(self):
        train_samples, val_samples, test_samples = [], [], []

        for model in self.models:
            model_views_df = self.df[(self.df['model'] == model) & (self.df['viewpoint'].isin(self.views))]
            grouped = model_views_df.groupby('viewpoint')
            view_images = {view: list(grouped.get_group(view)['image_path']) for view in self.views}
            min_view_images = min(len(view_images[view]) for view in self.views)

            random.seed(self.seed)
            
            for view in self.views:
                random.shuffle(view_images[view])
                view_images[view] = view_images[view][:min_view_images]

            train_paths = [view_images[v][:self.train_images] for v in self.views]
            remaining_paths = [view_images[v][self.train_images:] for v in self.views]

            if self.val_ratio + self.test_ratio > 0:
                n_remaining = len(remaining_paths[0])
                n_val = int(n_remaining * self.val_ratio)
                val_paths = [remaining_paths[v][:n_val] for v in range(len(self.views))]
                test_paths = [remaining_paths[v][n_val:] for v in range(len(self.views))]
            else:
                val_paths = [[] for _ in self.views]
                test_paths = remaining_paths

            for i in range(self.train_images):
                pair = [train_paths[j][i] for j in range(len(self.views))]
                train_samples.append((model, pair))
            
            for i in range(len(val_paths[0])):
                pair = [val_paths[j][i] for j in range(len(self.views))]
                val_samples.append((model, pair))

            for i in range(len(test_paths[0])):
                pair = [test_paths[j][i] for j in range(len(self.views))]
                test_samples.append((model, pair))

        train_dataset = SplitDataset(train_samples, self.label_encoder, self.transform, self.augment)
        val_dataset   = SplitDataset(val_samples,   self.label_encoder, self.transform, False)
        test_dataset  = SplitDataset(test_samples,  self.label_encoder, self.transform, False)
        
        return train_dataset, val_dataset, test_dataset
    
    def __str__(self):
        s = f"=== Dataset Overview ===\n"
        s += f"Views: {self.views}\n"
        s += f"Number of models: {self.num_models}\n"
        s += f"Train images per model per view: {self.train_images}\n"
        
        samples_per_model_val = np.array([len([p for m, p in self.val.samples if m == model]) for model in self.models])
        samples_per_model_test = np.array([len([p for m, p in self.test.samples if m == model]) for model in self.models])
               
        s += f"Val ratio = {self.val_ratio}, Test ratio = {self.test_ratio}\n"
        s += f"Total validation images: {len(self.val)}\n"
        s += f"Validation images per model per view: Mean {samples_per_model_val.mean()}, Std: {samples_per_model_val.std()}\n"
        s += f"Total test images: {len(self.test)}\n"
        s += f"Test images per model per view: Mean {samples_per_model_test.mean()}, Std: {samples_per_model_test.std()}\n"

        return s



class SplitDataset(Dataset):
    def __init__(self, samples, label_encoder, transform, augment):
        self.samples = samples
        self.label_encoder = label_encoder
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model, paths = self.samples[idx]
        images = [Image.open(p).convert("RGB") for p in paths]

        if self.augment:
            images = [F.hflip(img) if random.random() > 0.5 else img for img in images]

        if self.transform:
            images = [self.transform(img) for img in images]

        label = torch.tensor(self.label_encoder.transform([model])[0], dtype=torch.long)
        return {"images": images, "labels": label}