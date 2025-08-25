import pandas as pd
import torch
import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image


class UnseenDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            seen_dataset: Dataset,
            images: int
    ):
        self.df = df.copy()
        self.seen_dataset = seen_dataset
        self.images = images