"""
Test del módulo de data.

Testea la funcionalidad básica de DataFrameMaker, CarDataset y 
la creación DataFrames y Datasets estándar.
"""

import pytest
from src.data import DataFrameMaker, create_compcars_dataset
from src.data import CarDataset, create_car_dataset

BASE_PATH = '/Users/diegovega/Documents/Memoria/CompCars/dataset'


def test_data_frame_maker_creation():
    """Test creación básica de DataFrameMaker."""
    maker = DataFrameMaker(BASE_PATH)
    assert maker is not None
    assert hasattr(maker, 'base_path')
    assert hasattr(maker, 'image_folder')
    assert hasattr(maker, 'labels_folder')
    assert hasattr(maker, 'misc_folder')
    assert hasattr(maker, 'attributes_df')
    assert hasattr(maker, 'make_mapping')
    assert hasattr(maker, 'type_mapping')
    assert hasattr(maker, 'dataset_df')

def test_data_frame_maker_functios():
    """Test creación, guarado y summary del dataset"""
    

