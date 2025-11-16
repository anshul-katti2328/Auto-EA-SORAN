"""
Prediction Agent for Auto EA-SORAN System

"""

import asyncio
import zmq
import zmq.asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter, butter, sosfilt, sosfiltfilt
from sklearn.preprocessing import SplineTransformer
import shap
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import yaml
from dataclasses import dataclass, asdict
import pickle
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction_agent.log'),
        logging.StreamHandler()
    ]
)


# Ensure working directory is project root

@dataclass
class PredictionRequest:
    """Data structure for prediction requests"""
    request_id: str
    timestamp: float
    bs_id: str
    metrics: Dict[str, Any]
    config_params: Dict[str, Any]

@dataclass
class PredictionResponse:
    """Data structure for prediction responses"""
    request_id: str
    timestamp: float
    bs_id: str
    predicted_energy: float
    confidence_score: float
    feature_importance: Dict[str, float]
    explanation: str
    processing_time_ms: float

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def periodic_spline_transformer(period: int, n_knots: int = 12):
    """Create periodic spline transformer matching Farzi's implementation"""
    return SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)

class SwishActivation(nn.Module):
    """Swish activation function as used in Farzi's final model"""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

class EnergyPredictionANN(nn.Module):
    def __init__(self, input_dim: int, categorical_dims: Dict[str, int]):
        super().__init__()
        
        # Calculate total input dimension (continuous + categorical)
        self.total_input_dim = input_dim + len(categorical_dims)
        
        # Create network with correct input dimensions
        self.network = nn.Sequential(
            nn.Linear(self.total_input_dim, 128),
            SwishActivation(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            SwishActivation(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            SwishActivation(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            SwishActivation(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights(self.network)

    def forward(self, continuous_features: torch.Tensor, 
                categorical_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Handle categorical features
        if categorical_features:
            cat_features = torch.stack(list(categorical_features.values()), dim=1).float()
            all_features = torch.cat([continuous_features, cat_features], dim=1)
        else:
            all_features = continuous_features
        
        return self.network(all_features)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class FarziFeatureEngineer:
    """
    Exact feature engineering pipeline from Farzi's implementation
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.spline_transformer = None
        self.feature_names = []
        self.categorical_features = ['RUType', 'Mode', 'load_bin']
        self.is_fitted = False
        self.logger = logging.getLogger('FarziFeatureEngineer')

    def apply_farzi_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply exact preprocessing from Farzi notebook"""
        df_processed = df.copy()
        
        # Clean BS column
        if 'BS' in df_processed.columns:
            df_processed['BS'] = df_processed['BS'].str.replace(r'[a-zA-Z_]', '', regex=True).astype(int)
        
        # Clean categorical columns
        for col in ['RUType', 'Mode']:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].str.replace(r'[a-zA-Z]', '', regex=True).astype(int)
        
        return df_processed

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features exactly as Farzi did"""
        df_processed = df.copy()
        
        if 'Time' in df.columns:
            df_processed['Time'] = pd.to_datetime(df_processed['Time'])
            df_processed['day'] = df_processed['Time'].dt.day
            df_processed['weekday_number'] = df_processed['Time'].dt.weekday
            df_processed['hour'] = df_processed['Time'].dt.hour

            # Create periodic spline features (exactly as Farzi)
            hour_df = df_processed[['hour']].copy()
            if self.spline_transformer is None:
                self.spline_transformer = periodic_spline_transformer(24, n_knots=12)
                splines = self.spline_transformer.fit_transform(hour_df)
            else:
                splines = self.spline_transformer.transform(hour_df)
            
            splines_df = pd.DataFrame(splines, 
                                    columns=[f"hour_spline_{i}" for i in range(splines.shape[1])])
            df_processed = pd.concat([df_processed, splines_df], axis=1)

        return df_processed

    def create_lagged_features(self, df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
        """Create lagged features exactly as Farzi implementation"""
        df_processed = df.copy()
        
        # Sort by BS and Time for proper lagging
        if 'BS' in df_processed.columns and 'Time' in df_processed.columns:
            df_processed = df_processed.sort_values(['BS', 'Time'], ascending=True, ignore_index=True)

        # Create lags for specific columns (exactly as Farzi)
        lag_columns = ['load', 'Energy', 'ESMode1', 'ESMode2', 'ESMode3', 'ESMode6']
        all_shifts = list(np.arange(1, max_lag + 1))
        
        for shift_i in tqdm(all_shifts, desc="Creating lag features"):
            for col in lag_columns:
                if col in df_processed.columns:
                    if 'BS' in df_processed.columns:
                        df_processed[f'{col}_T-{shift_i}'] = df_processed.groupby(['BS'])[col].shift(shift_i)
                    else:
                        df_processed[f'{col}_T-{shift_i}'] = df_processed[col].shift(shift_i)

        # Time elapsed features (exactly as Farzi)
        for shift_i in tqdm(all_shifts, desc="Creating time elapsed features"):
            time_col = f'Time_T-{shift_i}'
            if time_col in df_processed.columns:
                df_processed[f'Time_T-{shift_i}_hours_elapsed'] = (
                    df_processed[time_col] - df_processed['Time']
                ).dt.total_seconds() / 3600
                del df_processed[time_col]  # Remove the time column after processing

        return df_processed

    def create_load_binning(self, df: pd.DataFrame, n_bins: int = 100) -> pd.DataFrame:
        """Create load binning exactly as Farzi implementation"""
        df_processed = df.copy()
        
        if 'load' in df_processed.columns:
            # Exact binning as in Farzi notebook
            bins = [round(i, 2) for i in list(np.arange(0, 1.01, 0.01))]
            labels = [f'{i}' for i in range(n_bins)]
            df_processed['load_bin'] = pd.cut(df_processed['load'], 
                                            bins=bins, 
                                            labels=labels)
            df_processed['load_bin'] = df_processed['load_bin'].astype(float).fillna(-1).astype(int)

        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Fit the complete Farzi feature engineering pipeline"""
        self.logger.info("Starting Farzi feature engineering pipeline...")
        
        # Apply preprocessing steps exactly as Farzi
        df_processed = self.apply_farzi_preprocessing(df)
        df_processed = self.create_temporal_features(df_processed)
        df_processed = self.create_lagged_features(df_processed)
        df_processed = self.create_load_binning(df_processed)

        self.logger.info(f"After feature engineering: {df_processed.shape}")

        # Identify features (exclude target and metadata)
        exclude_cols = {'Energy', 'Time', 'BS', 'CellName', 'ID', 'split'}
        all_features = [col for col in df_processed.columns if col not in exclude_cols]

        # Separate continuous and categorical features
        categorical_features = [col for col in all_features if col in self.categorical_features]
        continuous_features = [col for col in all_features if col not in self.categorical_features]
        self.feature_names = continuous_features + categorical_features

        # Handle missing values
        df_processed[self.feature_names] = df_processed[self.feature_names].fillna(0)

        # Process continuous features
        X_continuous = df_processed[continuous_features].values
        X_continuous_scaled = self.scaler.fit_transform(X_continuous)

        # Process categorical features
        X_categorical = {}
        for cat_feature in categorical_features:
            if cat_feature not in self.label_encoders:
                self.label_encoders[cat_feature] = LabelEncoder()
            X_categorical[cat_feature] = self.label_encoders[cat_feature].fit_transform(
                df_processed[cat_feature].astype(str)
            )

        self.is_fitted = True
        self.logger.info(f"Feature engineering completed. Continuous: {X_continuous_scaled.shape}, "
                        f"Categorical: {len(X_categorical)}")

        return X_continuous_scaled, X_categorical

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Transform new data using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")

        # Apply same transformations
        df_processed = self.apply_farzi_preprocessing(df)
        df_processed = self.create_temporal_features(df_processed)
        df_processed = self.create_lagged_features(df_processed)
        df_processed = self.create_load_binning(df_processed)

        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df_processed.columns:
                df_processed[feature] = 0

        # Separate features
        categorical_features = [col for col in self.feature_names if col in self.categorical_features]
        continuous_features = [col for col in self.feature_names if col not in self.categorical_features]

        # Process continuous features
        X_continuous = df_processed[continuous_features].fillna(0).values
        X_continuous_scaled = self.scaler.transform(X_continuous)

        # Process categorical features
        X_categorical = {}
        for cat_feature in categorical_features:
            if cat_feature in self.label_encoders:
                try:
                    X_categorical[cat_feature] = self.label_encoders[cat_feature].transform(
                        df_processed[cat_feature].fillna('unknown').astype(str)
                    )
                except ValueError:
                    # Handle unseen categories
                    X_categorical[cat_feature] = np.zeros(len(df_processed), dtype=int)

        return X_continuous_scaled, X_categorical

class FarziModelTrainer:
    """
    Complete training pipeline based on Farzi's notebook
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('FarziModelTrainer')
        self.feature_engineer = FarziFeatureEngineer()
        set_seed(42)  # Exact seed as Farzi

    def load_farzi_datasets(self, data_dir: str) -> pd.DataFrame:
        """Load and merge Farzi datasets exactly as in their notebook"""
        self.logger.info("Loading Farzi datasets...")
        
        # Load datasets with exact names from Farzi notebook
        df_train = pd.read_csv(f"{data_dir}/imgs_2023071012133740345.csv", parse_dates=['Time'])
        df_test = pd.read_csv(f"{data_dir}/imgs_202307101549519358.csv", parse_dates=['Time'])
        df_cell = pd.read_csv(f"{data_dir}/imgs_2023071012130978799.csv", parse_dates=['Time'])
        df_bs = pd.read_csv(f"{data_dir}/imgs_2023071012123392536.csv")

        # Merge exactly as Farzi did
        df_features = df_cell.merge(df_bs, on=['BS', 'CellName'], how='outer')
        df_features = df_features[df_features['CellName'] == 'Cell0'].reset_index(drop=True)

        df_test['split'] = 'test'
        df_train['split'] = 'train'
        df_total = pd.concat([df_train, df_test], ignore_index=True)
        df_total = df_total.merge(df_features, on=['BS', 'Time'], how='left')

        # Create ID exactly as Farzi
        df_total['ID'] = df_total['Time'].astype(str) + "_" + df_total['BS']

        self.logger.info(f"Loaded datasets with shape: {df_total.shape}")
        return df_total

    def train_model(self, data_dir: str, save_dir: str = ".venv/models/"):
        """Complete training pipeline matching Farzi's approach"""
        self.logger.info("Starting Farzi model training...")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and process data
        df = self.load_farzi_datasets(data_dir)
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=['Energy']).reset_index(drop=True)
        self.logger.info(f"Clean data shape: {df_clean.shape}")

        # Feature engineering
        X_continuous, X_categorical = self.feature_engineer.fit_transform(df_clean)
        y = df_clean['Energy'].values

        # Split data exactly as Farzi (train/test split based on 'split' column)
        train_size = int(0.8 * len(df_clean))
        indices = np.random.permutation(len(df_clean))
        train_mask = np.zeros(len(df_clean), dtype=bool)
        train_mask[indices[:train_size]] = True

        # Apply split
        X_train_cont = X_continuous[train_mask]
        X_val_cont = X_continuous[~train_mask]
        X_train_cat = {k: v[train_mask] for k, v in X_categorical.items()}
        X_val_cat = {k: v[~train_mask] for k, v in X_categorical.items()}
        y_train, y_val = y[train_mask], y[~train_mask]

        self.logger.info(f"Train set: {len(y_train)}, Val set: {len(y_val)}")

        # Initialize model with correct dimensions
        categorical_dims = {name: len(np.unique(data)) for name, data in X_categorical.items()}
        model = EnergyPredictionANN(X_continuous.shape[1], categorical_dims)
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_cont).to(device)
        X_val_tensor = torch.FloatTensor(X_val_cont).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)

        X_train_cat_tensors = {k: torch.LongTensor(v).to(device) for k, v in X_train_cat.items()}
        X_val_cat_tensors = {k: torch.LongTensor(v).to(device) for k, v in X_val_cat.items()}

        # Training loop with exact parameters as Farzi
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        epochs = 100  # Matching Farzi's typical training

        self.logger.info("Starting model training...")
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor, X_train_cat_tensors)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor, X_val_cat_tensors)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'categorical_dims': categorical_dims,
                    'epoch': epoch,
                    'val_loss': val_loss.item(),
                    'train_loss': loss.item()
                }, f"{save_dir}/energy_prediction_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss {loss:.4f}, Val Loss {val_loss:.4f}")

        # Save feature engineer
        with open(f"{save_dir}/feature_engineer.pkl", 'wb') as f:
            pickle.dump(self.feature_engineer, f)

        self.logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        return best_val_loss

class PredictionAgent:
    """
    Enhanced Prediction Agent with Farzi training and real-time deployment
    """
    def __init__(self, config_path: str = "prediction_agent_config.yaml"):
        self.logger = logging.getLogger('PredictionAgent')
        self.config = self.load_config(config_path)
        self.agent_id = self.config['agent']['id']
        self.device = torch.device(self.config['model']['device'])

        # Initialize components
        self.feature_engineer = FarziFeatureEngineer()
        self.model = None
        self.explainer = None
        self.is_model_loaded = False

        # Check if training is needed
        self.should_train = not self._check_model_exists()
        if self.should_train:
            self.logger.info("Pre-trained model not found. Training will be required.")

        # ZeroMQ setup for agentic communication
        self.context = zmq.asyncio.Context()
        self.request_socket = None
        self.response_socket = None

        # Performance tracking
        self.metrics = {
            'requests_processed': 0,
            'total_processing_time': 0,
            'errors': 0,
            'start_time': time.time()
        }

        self.logger.info(f"PredictionAgent {self.agent_id} initialized")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'agent': {
                'id': 'prediction_agent_001',
                'name': 'Farzi Enhanced Prediction Agent'
            },
            'model': {
                'device': 'cpu',
                'input_dim': 99
            },
            'training': {
                'data_dir': '.venv/farzi_data/',
                'max_epochs': 100,
                'early_stopping_patience': 10,
                'learning_rate': 0.001
            },
            'communication': {
                'request_port': 5555,
                'response_port': 5556,
                'optimization_agent_port': 5556
            },
            'paths': {
                'model_checkpoint': '.venv/models/energy_prediction_model.pth',
                'feature_engineer': '.venv/models/feature_engineer.pkl'
            }
        }

    def _check_model_exists(self) -> bool:
        """Check if trained model files exist"""
        model_path = self.config['paths']['model_checkpoint']
        feature_path = self.config['paths']['feature_engineer']
        return os.path.exists(model_path) and os.path.exists(feature_path)

    async def initialize_communication(self):
        """Initialize ZeroMQ communication sockets"""
        try:
            # Socket for receiving prediction requests
            self.request_socket = self.context.socket(zmq.REP)
            self.request_socket.bind(f"tcp://*:{self.config['communication']['request_port']}")

            # Socket for sending responses to optimization agent
            self.response_socket = self.context.socket(zmq.PUSH)
            self.response_socket.connect(f"tcp://localhost:{self.config['communication']['optimization_agent_port']}")

            self.logger.info("Communication channels initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def load_model(self):
        """Load pre-trained model or train new model if needed"""
        
        # If training is needed, do it first
        if self.should_train:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir_config = self.config.get('training', {}).get('data_dir', 'farzi_data')
            data_dir = os.path.abspath(os.path.join(project_root, data_dir_config))
            if os.path.exists(data_dir):
                trainer = FarziModelTrainer(self.config)
                trainer.train_model(data_dir, os.path.dirname(self.config['paths']['model_checkpoint']))
                self.logger.info("Model training completed")
            else:
                raise FileNotFoundError(f"Training data directory not found: {data_dir}")

        try:
            # Load feature engineer
            with open(self.config['paths']['feature_engineer'], 'rb') as f:
                self.feature_engineer = pickle.load(f)

            # Load model
            checkpoint = torch.load(self.config['paths']['model_checkpoint'],
                                  map_location=self.device)
            categorical_dims = checkpoint['categorical_dims']
            
            self.model = EnergyPredictionANN(
                len(self.feature_engineer.feature_names) - len(self.feature_engineer.categorical_features),
                categorical_dims
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_model_loaded = True
            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_energy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make energy predictions with confidence scoring"""
        if not self.is_model_loaded:
            raise ValueError("Model not loaded")

        # Feature engineering
        X_continuous, X_categorical = self.feature_engineer.transform(df)

        # Convert to tensors
        X_continuous_tensor = torch.FloatTensor(X_continuous).to(self.device)
        X_categorical_tensors = {
            name: torch.LongTensor(data).to(self.device)
            for name, data in X_categorical.items()
        }

        # Prediction with Monte Carlo Dropout for confidence
        predictions = []
        self.model.train()  # Enable dropout for MC sampling
        with torch.no_grad():
            for _ in range(100):  # MC samples
                pred = self.model(X_continuous_tensor, X_categorical_tensors)
                predictions.append(pred.cpu().numpy().flatten())

        self.model.eval()  # Return to eval mode
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        confidence = 1.0 / (1.0 + np.std(predictions, axis=0))  # Inverse of uncertainty

        return mean_prediction, confidence

    def generate_explanation(self, df: pd.DataFrame, prediction: float) -> Dict[str, Any]:
        """Generate SHAP-based explanations"""
        try:
            if self.explainer is None:
                # Initialize SHAP explainer
                X_continuous, X_categorical = self.feature_engineer.transform(df)
                background_indices = np.random.choice(len(X_continuous),
                                                    min(100, len(X_continuous)),
                                                    replace=False)

                X_bg_cont = torch.FloatTensor(X_continuous[background_indices]).to(self.device)
                X_bg_cat = {
                    name: torch.LongTensor(data[background_indices]).to(self.device)
                    for name, data in X_categorical.items()
                }

                def model_wrapper(x_cont):
                    x_cont_tensor = torch.FloatTensor(x_cont).to(self.device)
                    with torch.no_grad():
                        return self.model(x_cont_tensor, X_bg_cat).cpu().numpy()

                self.explainer = shap.KernelExplainer(model_wrapper, X_continuous[background_indices])

            # Generate explanations
            X_continuous, _ = self.feature_engineer.transform(df)
            shap_values = self.explainer.shap_values(X_continuous[:1])

            # Create feature importance dictionary
            continuous_features = [f for f in self.feature_engineer.feature_names
                                 if f not in self.feature_engineer.categorical_features]
            feature_importance = {
                feature: float(importance)
                for feature, importance in zip(continuous_features, shap_values[0])
            }

            # Generate textual explanation
            top_features = sorted(feature_importance.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:3]
            explanation = f"Energy prediction: {prediction:.2f}W. "
            explanation += "Key factors: "
            for feature, importance in top_features:
                direction = "increases" if importance > 0 else "decreases"
                explanation += f"{feature} {direction} energy by {abs(importance):.3f}; "

            return {
                'feature_importance': feature_importance,
                'explanation': explanation.rstrip('; '),
                'top_features': top_features
            }

        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
            return {
                'feature_importance': {},
                'explanation': f"Prediction: {prediction:.2f}W. Explanation unavailable.",
                'top_features': []
            }

    async def process_prediction_request(self, request: PredictionRequest) -> PredictionResponse:
        """Process a single prediction request"""
        start_time = time.time()
        try:
            # Convert request data to DataFrame
            df_data = {**request.metrics, **request.config_params}
            df = pd.DataFrame([df_data])

            # Make prediction
            predictions, confidence = self.predict_energy(df)
            prediction = float(predictions[0])
            confidence_score = float(confidence[0])

            # Generate explanation
            explanation_data = self.generate_explanation(df, prediction)

            # Create response
            processing_time = (time.time() - start_time) * 1000
            response = PredictionResponse(
                request_id=request.request_id,
                timestamp=time.time(),
                bs_id=request.bs_id,
                predicted_energy=prediction,
                confidence_score=confidence_score,
                feature_importance=explanation_data['feature_importance'],
                explanation=explanation_data['explanation'],
                processing_time_ms=processing_time
            )

            # Update metrics
            self.metrics['requests_processed'] += 1
            self.metrics['total_processing_time'] += processing_time

            self.logger.info(f"Processed prediction for {request.bs_id}: {prediction:.2f}W "
                           f"(confidence: {confidence_score:.3f})")

            return response

        except Exception as e:
            self.logger.error(f"Failed to process prediction request: {e}")
            self.metrics['errors'] += 1

            # Return error response
            return PredictionResponse(
                request_id=request.request_id,
                timestamp=time.time(),
                bs_id=request.bs_id,
                predicted_energy=-1.0,
                confidence_score=0.0,
                feature_importance={},
                explanation=f"Prediction failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

    async def run(self):
        """Main agent execution loop"""
        self.logger.info(f"Starting PredictionAgent {self.agent_id}")
        
        try:
            # Initialize
            await self.initialize_communication()
            self.load_model()
            self.logger.info("PredictionAgent ready for requests")

            # Main processing loop
            while True:
                try:
                    # Wait for prediction request
                    message = await self.request_socket.recv_json(zmq.NOBLOCK)
                    request = PredictionRequest(**message)

                    # Process request
                    response = await self.process_prediction_request(request)

                    # Send response back to requester
                    await self.request_socket.send_json(asdict(response))

                    # Also send to optimization agent
                    await self.response_socket.send_json(asdict(response))

                except zmq.Again:
                    # No message available, continue
                    await asyncio.sleep(0.01)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)

        finally:
            # Cleanup
            if self.request_socket:
                self.request_socket.close()
            if self.response_socket:
                self.response_socket.close()
            self.context.term()
            self.logger.info("PredictionAgent stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Farzi Enhanced Prediction Agent')
    parser.add_argument('--config', default='prediction_agent_config.yaml',
                       help='Configuration file path')
    args = parser.parse_args()

    agent = PredictionAgent(args.config)
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nShutting down PredictionAgent...")
