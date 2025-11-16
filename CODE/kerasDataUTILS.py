"""
Data utilities for Keras model training
Converts FarziFeatureEngineer output to Keras-compatible format
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import pickle
import tensorflow as tf

def prepare_keras_inputs(
    X_continuous: np.ndarray,
    X_categorical: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Convert FarziFeatureEngineer output to Keras model input format
    
    Args:
        X_continuous: Continuous features array (N, n_continuous)
        X_categorical: Dict of categorical features {name: (N,)}
    
    Returns:
        Dict of inputs ready for Keras model
    """
    inputs = {'continuous': X_continuous.astype(np.float32)}
    
    # Reshape categorical features from (N,) to (N, 1)
    for name, data in X_categorical.items():
        inputs[name] = data.reshape(-1, 1).astype(np.int32)
    
    return inputs

def prepare_keras_target(y: np.ndarray) -> np.ndarray:
    """
    Prepare target variable for Keras
    
    Args:
        y: Target array (N,) or (N, 1)
    
    Returns:
        Reshaped target (N, 1) as float32
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y.astype(np.float32)

def split_data(
    X_continuous: np.ndarray,
    X_categorical: Dict[str, np.ndarray],
    y: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Split data into train and validation sets
    
    Args:
        X_continuous: Continuous features
        X_categorical: Categorical features dict
        y: Target variable
        train_ratio: Proportion of data for training
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    n_samples = len(X_continuous)
    n_train = int(n_samples * train_ratio)
    
    # Create indices
    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Split continuous
    X_train_cont = X_continuous[train_idx]
    X_val_cont = X_continuous[val_idx]
    
    # Split categorical
    X_train_cat = {name: data[train_idx] for name, data in X_categorical.items()}
    X_val_cat = {name: data[val_idx] for name, data in X_categorical.items()}
    
    # Split target
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    # Prepare for Keras
    X_train = prepare_keras_inputs(X_train_cont, X_train_cat)
    X_val = prepare_keras_inputs(X_val_cont, X_val_cat)
    y_train = prepare_keras_target(y_train)
    y_val = prepare_keras_target(y_val)
    
    return X_train, X_val, y_train, y_val

class KerasDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for large datasets
    """
    def __init__(
        self,
        X_continuous: np.ndarray,
        X_categorical: Dict[str, np.ndarray],
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        self.X_continuous = X_continuous
        self.X_categorical = X_categorical
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X_continuous))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X_continuous) / self.batch_size))
    
    def __getitem__(self, index):
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        X_batch = prepare_keras_inputs(
            self.X_continuous[batch_indices],
            {name: data[batch_indices] for name, data in self.X_categorical.items()}
        )
        y_batch = prepare_keras_target(self.y[batch_indices])
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

if __name__ == "__main__":
    print("Testing data utilities...")
    print("=" * 80)
    
    # Test with dummy data (no feature engineer needed for testing utilities)
    n_samples = 100
    X_cont = np.random.randn(n_samples, 46)
    X_cat = {
        'RUType': np.random.randint(0, 12, n_samples),
        'Mode': np.random.randint(0, 2, n_samples),
        'load_bin': np.random.randint(0, 101, n_samples)
    }
    y = np.random.randn(n_samples)
    
    # Test prepare_keras_inputs
    inputs = prepare_keras_inputs(X_cont, X_cat)
    print(f"\n✓ Keras inputs prepared:")
    for name, data in inputs.items():
        print(f"  - {name}: shape {data.shape}, dtype {data.dtype}")
    
    # Test split
    X_train, X_val, y_train, y_val = split_data(X_cont, X_cat, y)
    print(f"\n✓ Data split:")
    print(f"  - Train samples: {len(y_train)}")
    print(f"  - Val samples: {len(y_val)}")
    
    # Test data generator
    print(f"\n✓ Testing data generator:")
    gen = KerasDataGenerator(X_cont, X_cat, y, batch_size=16)
    print(f"  - Number of batches: {len(gen)}")
    X_batch, y_batch = gen[0]
    print(f"  - First batch X keys: {list(X_batch.keys())}")
    print(f"  - First batch y shape: {y_batch.shape}")
    
    print("\n✓ All data utilities working correctly!")
    print("\n" + "=" * 80)
    print("NOTE: To test with real feature engineer, use:")
    print("  from featureENGINEER import FarziFeatureEngineer")
    print("  fe = FarziFeatureEngineer()")
    print("=" * 80)