"""
Pruebas unitarias para el módulo de datos
"""

import pytest
import tempfile
import os
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PairedImageDataset


@pytest.fixture
def temp_dataset_dir():
    """Crea directorios temporales con imágenes de prueba"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, 'input')
        gt_dir = os.path.join(tmpdir, 'gt')
        
        os.makedirs(input_dir)
        os.makedirs(gt_dir)
        
        # Crear imágenes de prueba
        for i in range(3):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(input_dir, f'img_{i}.jpg'), img)
            cv2.imwrite(os.path.join(gt_dir, f'img_{i}.jpg'), img)
        
        yield input_dir, gt_dir


def test_dataset_load(temp_dataset_dir):
    """Prueba carga del dataset"""
    input_dir, gt_dir = temp_dataset_dir
    
    dataset = PairedImageDataset(input_dir, gt_dir)
    
    assert len(dataset) == 3
    assert dataset.img_files == ['img_0.jpg', 'img_1.jpg', 'img_2.jpg']


def test_dataset_getitem(temp_dataset_dir):
    """Prueba obtener item del dataset"""
    input_dir, gt_dir = temp_dataset_dir
    
    dataset = PairedImageDataset(input_dir, gt_dir)
    item = dataset[0]
    
    assert 'input' in item
    assert 'gt' in item
    assert 'filename' in item
    assert item['input'].shape == (256, 256, 3)
    assert item['gt'].shape == (256, 256, 3)
    assert item['filename'] == 'img_0.jpg'


def test_dataset_values_normalized(temp_dataset_dir):
    """Prueba que los valores estén normalizados"""
    input_dir, gt_dir = temp_dataset_dir
    
    dataset = PairedImageDataset(input_dir, gt_dir)
    item = dataset[0]
    
    assert item['input'].min() >= 0.0
    assert item['input'].max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
