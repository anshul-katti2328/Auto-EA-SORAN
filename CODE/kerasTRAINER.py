"""
Keras Model Trainer for Energy Prediction
Trains the model using FarziFeatureEngineer preprocessed data
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import SplineTransformer
from scipy.signal import savgol_filter

# Import our custom modules
from kerasModelARCH import build_energy_prediction_model, compile_model
from kerasDataUTILS import prepare_keras_inputs, split_data, KerasDataGenerator
from featureENGINEER import FarziFeatureEngineer  # ← ADD THIS


# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keras_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ==================== Training Configuration ====================
class TrainingConfig:
    """Training hyperparameters"""
    def __init__(self):
        # Paths
        self.data_dir = Path("../archive")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Data files
        self.train_file = "merged_training_data.csv"
        self.test_file = "imgs_202307101549519358.csv"
        
        # Model architecture
        self.continuous_dim = 46
        self.categorical_dims = {'RUType': 12, 'Mode': 2, 'load_bin': 101}
        self.hidden_dims = [128, 256, 128, 64]
        self.dropout_rate = 0.2
        
        # Training
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 0.001
        self.train_split = 0.8
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 5
        
        # Output
        self.model_save_path = self.model_dir / "energy_prediction_keras.keras"
        self.feature_engineer_path = self.model_dir / "feature_engineer_keras.pkl"
        self.history_path = self.model_dir / "training_history.json"


# ==================== Data Loading ====================
def load_and_prepare_data(config: TrainingConfig) -> Tuple:
    """Load and prepare training data"""
    logger.info("=" * 80)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("=" * 80)
    
    # Load CSV
    train_path = config.data_dir / config.train_file
    logger.info(f"Loading data from: {train_path}")
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    df = pd.read_csv(train_path)
    logger.info(f"✓ Loaded {len(df)} rows")
    
    # Basic cleaning
    logger.info("Cleaning data...")
    df_clean = df.dropna(subset=['Energy'])
    df_clean = df_clean[df_clean['Energy'] > 0]
    logger.info(f"✓ After cleaning: {len(df_clean)} rows")
    
    # Feature engineering
    logger.info("Applying feature engineering...")
    feature_engineer = FarziFeatureEngineer()
    X_continuous, X_categorical, y = feature_engineer.fit_transform(df_clean, target_col='Energy')
    config.continuous_dim = X_continuous.shape[1]
    config.categorical_dims = {
        name: len(feature_engineer.label_encoders[name].classes_)
        for name in X_categorical.keys()
    }
    
    logger.info(f"\n📊 DATA STATISTICS:")
    logger.info(f"   Continuous features: {config.continuous_dim}")
    logger.info(f"   Categorical features:")
    for name, dim in config.categorical_dims.items():
        logger.info(f"      - {name}: {dim} classes")

    # Save feature engineer
    with open(config.feature_engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    logger.info(f"✓ Feature engineer saved to: {config.feature_engineer_path}")
    
    # Split data
    logger.info(f"Splitting data (train: {config.train_split:.0%}, val: {1-config.train_split:.0%})...")
    X_train, X_val, y_train, y_val = split_data(
        X_continuous, X_categorical, y,
        train_ratio=config.train_split,
        shuffle=True,
        random_seed=RANDOM_SEED
    )
    
    logger.info(f"✓ Training samples: {len(y_train)}")
    logger.info(f"✓ Validation samples: {len(y_val)}")
    logger.info(f"✓ Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X_train, X_val, y_train, y_val, feature_engineer


# ==================== Training Callbacks ====================
def create_callbacks(config: TrainingConfig) -> list:
    """Create Keras callbacks"""
    callbacks = []
    
    # Model checkpoint (save best model)
    checkpoint_path = str(config.model_save_path).replace('.keras', '_checkpoint.keras')
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=config.reduce_lr_patience,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard)
    
    return callbacks


# ==================== Main Training Function ====================
def train_model(config: TrainingConfig):
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("KERAS ENERGY PREDICTION MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    X_train, X_val, y_train, y_val, feature_engineer = load_and_prepare_data(config)
    
    # Build model
    logger.info("\n" + "=" * 80)
    logger.info("BUILDING MODEL")
    logger.info("=" * 80)
    
    model = build_energy_prediction_model(
        continuous_dim=config.continuous_dim,
        categorical_dims=config.categorical_dims,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate
    )
    
    model = compile_model(model, learning_rate=config.learning_rate)
    
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80)
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)
    
    model.save(config.model_save_path)
    logger.info(f"✓ Model saved to: {config.model_save_path}")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'config': {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'train_samples': len(y_train),
            'val_samples': len(y_val)
        }
    }
    
    with open(config.history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"✓ Training history saved to: {config.history_path}")
    
    # Final evaluation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)
    
    val_loss, val_mae, val_rmse = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation Loss (MSE): {val_loss:.4f}")
    logger.info(f"Validation MAE: {val_mae:.4f}")
    logger.info(f"Validation RMSE: {val_rmse:.4f}")
    
    # Sample predictions
    sample_preds = model.predict(X_val, verbose=0)[:5].flatten()
    sample_actual = y_val[:5].flatten()
    
    logger.info("\nSample Predictions vs Actual:")
    for i, (pred, actual) in enumerate(zip(sample_preds, sample_actual)):
        logger.info(f"  {i+1}. Predicted: {pred:.2f}, Actual: {actual:.2f}, Error: {abs(pred-actual):.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        config = TrainingConfig()
        train_model(config)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)