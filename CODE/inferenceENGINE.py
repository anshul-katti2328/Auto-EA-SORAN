"""
Keras Inference Engine for Energy Prediction with SHAP Explainability
Load trained model and make predictions on new data
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import logging
import warnings
import json
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# SHAP for explainability
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

from kerasDataUTILS import prepare_keras_inputs
from kerasModelARCH import Swish

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KerasEnergyPredictor:
    """
    Production-ready inference engine for energy prediction with SHAP explainability
    """
    
    def __init__(
        self,
        model_path: str = "models/energy_prediction_keras.keras",
        feature_engineer_path: str = "models/feature_engineer_keras.pkl",
        enable_shap: bool = False
    ):
        """
        Initialize predictor with trained model and feature engineer
        
        Args:
            model_path: Path to trained Keras model
            feature_engineer_path: Path to pickled feature engineer
            enable_shap: Whether to enable SHAP explainability
        """
        self.model_path = Path(model_path)
        self.feature_engineer_path = Path(feature_engineer_path)
        self.enable_shap = enable_shap
        
        logger.info("=" * 80)
        logger.info("INITIALIZING KERAS ENERGY PREDICTOR")
        logger.info("=" * 80)
        
        # Load model with custom objects
        logger.info(f"Loading model from: {self.model_path}")
        custom_objects = {'Swish': Swish}
        self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
        logger.info(f"✓ Model loaded: {self.model.count_params():,} parameters")
        
        # Load feature engineer
        logger.info(f"Loading feature engineer from: {self.feature_engineer_path}")
        with open(self.feature_engineer_path, 'rb') as f:
            self.feature_engineer = pickle.load(f)
        logger.info(f"✓ Feature engineer loaded")
        logger.info(f"   - Continuous features: {len(self.feature_engineer.continuous_feature_names)}")
        logger.info(f"   - Categorical features: {len(self.feature_engineer.categorical_features)}")
        
        # SHAP components
        self.shap_explainer = None
        self.background_data = None
        
        if self.enable_shap:
            logger.info("SHAP explainability: ENABLED (will initialize on first use)")
        else:
            logger.info("SHAP explainability: DISABLED")
        
        logger.info("=" * 80)
        logger.info("✓ PREDICTOR READY")
        logger.info("=" * 80)
    
    def predict(
        self, 
        df: pd.DataFrame,
        return_confidence: bool = False
    ) -> np.ndarray:
        """
        Predict energy consumption for input data
        
        Args:
            df: DataFrame with same structure as training data
                Must have columns: Time, BS, load, RUType, Mode, etc.
            return_confidence: If True, return (predictions, confidence_intervals)
        
        Returns:
            predictions: Array of predicted energy values
            OR (predictions, confidence) if return_confidence=True
        """
        logger.info(f"Making predictions for {len(df)} samples...")
        
        # Feature engineering
        X_continuous, X_categorical = self.feature_engineer.transform(df)
        
        # Prepare inputs
        keras_inputs = prepare_keras_inputs(X_continuous, X_categorical)
        
        # Predict
        predictions = self.model.predict(keras_inputs, verbose=0)
        predictions = predictions.flatten()
        
        logger.info(f"✓ Predictions complete")
        logger.info(f"   - Mean: {predictions.mean():.2f}")
        logger.info(f"   - Std: {predictions.std():.2f}")
        logger.info(f"   - Min: {predictions.min():.2f}")
        logger.info(f"   - Max: {predictions.max():.2f}")
        
        if return_confidence:
            # ✅ MC DROPOUT: Enable dropout during inference for uncertainty estimation
            # Run multiple forward passes with dropout active to get prediction distribution
            n_iterations = 10  # Number of stochastic forward passes
            
            logger.info(f"📊 Running MC Dropout with {n_iterations} iterations...")
            
            # Store predictions from each forward pass
            mc_predictions = []
            
            for i in range(n_iterations):
                # Make prediction with dropout enabled (training=True)
                mc_pred = self.model(keras_inputs, training=True)
                mc_predictions.append(mc_pred.numpy().flatten())
            
            # Stack predictions: shape (n_iterations, n_samples)
            mc_predictions = np.array(mc_predictions)
            
            # Calculate mean and std across iterations for each sample
            mc_mean = mc_predictions.mean(axis=0)  # Mean prediction
            mc_std = mc_predictions.std(axis=0)    # Uncertainty (std deviation)
            
            # Convert uncertainty to confidence score using coefficient of variation
            # Lower relative uncertainty = higher confidence
            relative_uncertainty = mc_std / (np.abs(mc_mean) + 1e-6)  # CV = std/mean, avoid division by zero
            
            # Convert to confidence using exponential decay
            # Low relative uncertainty → high confidence
            # High relative uncertainty → low confidence
            confidence = np.exp(-relative_uncertainty * 2)  # Scale factor of 2 for sensitivity
            
            # Clip to reasonable range [0.5, 0.99]
            # Minimum 0.5 because any prediction has some base confidence
            # Maximum 0.99 to indicate some inherent uncertainty always exists
            confidence = np.clip(confidence, 0.5, 0.99)
            
            logger.info(f"✓ MC Dropout complete:")
            logger.info(f"   - Mean prediction: {mc_mean.mean():.2f}W")
            logger.info(f"   - Avg uncertainty (std): {mc_std.mean():.2f}W")
            logger.info(f"   - Relative uncertainty (CV): {relative_uncertainty.mean():.3f}")
            logger.info(f"   - Avg confidence: {confidence.mean():.3f}")
            logger.info(f"   - Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
            
            # Use MC Dropout mean as final prediction instead of single pass
            predictions = mc_mean
            
            return predictions, confidence
        
        return predictions
    
    def predict_single(
        self,
        time: str,
        bs_id: str,
        load: float,
        ru_type: str,
        mode: str,
        **kwargs
    ) -> float:
        """
        Predict energy for a single observation
        
        Args:
            time: Timestamp (e.g., "2023-07-10 12:00:00")
            bs_id: Base station ID (e.g., "B_0")
            load: Traffic load
            ru_type: Radio unit type
            mode: Operational mode
            **kwargs: Additional features (Frequency, Bandwidth, etc.)
        
        Returns:
            Predicted energy value
        """
        # Create single-row DataFrame
        data = {
            'Time': [time],
            'BS': [bs_id],
            'load': [load],
            'RUType': [ru_type],
            'Mode': [mode],
            **kwargs
        }
        df = pd.DataFrame(data)
        
        prediction = self.predict(df)[0]
        return prediction
    
    def predict_batch_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Predict for entire CSV file
        
        Args:
            csv_path: Path to input CSV
            output_path: Optional path to save predictions
        
        Returns:
            DataFrame with predictions added
        """
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        predictions = self.predict(df)
        df['Energy_Predicted'] = predictions
        
        if 'Energy' in df.columns:
            df['Error'] = np.abs(df['Energy'] - df['Energy_Predicted'])
            df['Error_Percent'] = (df['Error'] / df['Energy'] * 100)
            
            logger.info("\n" + "=" * 80)
            logger.info("PREDICTION QUALITY METRICS")
            logger.info("=" * 80)
            logger.info(f"MAE: {df['Error'].mean():.4f}")
            logger.info(f"RMSE: {np.sqrt((df['Error']**2).mean()):.4f}")
            logger.info(f"MAPE: {df['Error_Percent'].mean():.2f}%")
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Predictions saved to: {output_path}")
        
        return df
    
    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = 'Energy'
    ) -> Dict[str, float]:
        """
        Evaluate model on labeled data
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
        
        Returns:
            Dictionary of metrics
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        predictions = self.predict(df)
        actuals = df[target_col].values
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'samples': len(df)
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        for key, value in metrics.items():
            if key != 'samples':
                logger.info(f"{key.upper():10s}: {value:.4f}")
            else:
                logger.info(f"{key.upper():10s}: {value}")
        
        return metrics
    
    # ==================== SHAP EXPLAINABILITY METHODS ====================
    
    # Replace the initialize_shap method (around line 250):

    # Replace the initialize_shap method (around line 250):

    # Replace the initialize_shap method:

    def initialize_shap(
        self, 
        background_samples: int = 100,
        background_data_path: str = "../archive/merged_training_data.csv"
    ):
        """
        Initialize SHAP explainer with background dataset
        
        Args:
            background_samples: Number of samples for SHAP background
            background_data_path: Path to training data for background
        """
        if self.shap_explainer is not None:
            logger.info("SHAP explainer already initialized")
            return
        
        logger.info("=" * 80)
        logger.info("INITIALIZING SHAP EXPLAINER")
        logger.info("=" * 80)
        logger.info(f"Loading {background_samples} background samples from: {background_data_path}")
        
        try:
            # Load background data
            train_data = pd.read_csv(background_data_path, nrows=background_samples)
            X_continuous_bg, X_categorical_bg = self.feature_engineer.transform(train_data)
            
            # Store background data (continuous features only for SHAP)
            self.background_data = {
                'continuous': X_continuous_bg,
                'categorical': X_categorical_bg,
            }
            
            logger.info(f"Background data shape: {X_continuous_bg.shape}")
            logger.info(f"Using KernelExplainer (model-agnostic, works with any model)")
            
            # Create a prediction function that wraps the model
            def model_predict(X_cont):
                """Wrapper function for SHAP that takes only continuous features"""
                # Use mean values for categorical features from background
                batch_size = X_cont.shape[0]
                
                # Create dummy categorical features (use mode from background)
                X_cat_dummy = {}
                for cat_feat, cat_values in X_categorical_bg.items():
                    # Use the most common value
                    mode_val = np.bincount(cat_values).argmax()
                    X_cat_dummy[cat_feat] = np.full(batch_size, mode_val, dtype=np.int32)
                
                # Prepare keras inputs
                keras_inputs = prepare_keras_inputs(X_cont, X_cat_dummy)
                
                # Predict
                preds = self.model.predict(keras_inputs, verbose=0)
                return preds.flatten()
            
            # Create SHAP KernelExplainer
            # Use a subset of background for speed
            bg_subset = X_continuous_bg[:min(50, len(X_continuous_bg))]
            
            self.shap_explainer = shap.KernelExplainer(
                model_predict,
                bg_subset
            )
            
            logger.info("✓ SHAP explainer initialized successfully")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            import traceback
            traceback.print_exc()
            self.shap_explainer = None
            raise

    def explain_prediction(
        self,
        df: pd.DataFrame,
        sample_idx: int = 0,
        plot: bool = True,
        save_path: Optional[str] = None,
        n_samples: int = 3
    ) -> Dict[str, Any]:
        """
        Explain predictions using SHAP values
        
        Args:
            df: Input data
            sample_idx: Index of sample to explain in detail
            plot: Whether to generate plots
            save_path: Path to save plots (default: shap_explanation.png)
            n_samples: Number of samples to explain (KernelExplainer is slow, use small number)
        
        Returns:
            Dictionary with SHAP values, feature names, and explanations
        """
        if not self.enable_shap:
            logger.warning("SHAP is disabled. Enable with enable_shap=True")
            return {}
        
        if self.shap_explainer is None:
            logger.info("SHAP explainer not initialized, initializing now...")
            self.initialize_shap()
        
        # Limit samples for KernelExplainer (it's slow)
        df_subset = df.head(n_samples)
        
        logger.info("=" * 80)
        logger.info(f"EXPLAINING PREDICTIONS FOR {len(df_subset)} SAMPLES")
        logger.info(f"(KernelExplainer is slow, using subset for speed)")
        logger.info("=" * 80)
        
        # Transform data
        X_continuous, X_categorical = self.feature_engineer.transform(df_subset)
        
        # Get feature names (continuous only)
        feature_names = self.feature_engineer.continuous_feature_names.copy()
        
        # Calculate SHAP values (this will be slow)
        logger.info("Calculating SHAP values (this may take several minutes)...")
        logger.info(f"Progress: Processing {len(X_continuous)} samples...")
        
        try:
            shap_values = self.shap_explainer.shap_values(
                X_continuous,
                nsamples='auto',  # Adaptive sampling
                l1_reg='aic'      # Automatic regularization
            )
            
            # Handle output format
            if isinstance(shap_values, list):
                shap_values_array = shap_values[0]
            else:
                shap_values_array = shap_values
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
        # Ensure shapes match
        if shap_values_array.shape[1] != len(feature_names):
            logger.warning(f"Shape mismatch: SHAP values have {shap_values_array.shape[1]} features, "
                          f"but we have {len(feature_names)} feature names. Using first {shap_values_array.shape[1]} names.")
            feature_names = feature_names[:shap_values_array.shape[1]]
        
        # Calculate feature importance (mean absolute SHAP)
        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        logger.info("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
        logger.info(importance_df.head(10).to_string(index=False))
        
        # Get predictions
        keras_inputs = prepare_keras_inputs(X_continuous, X_categorical)
        predictions = self.model.predict(keras_inputs, verbose=0).flatten()
        
        # Detailed explanation for specific sample
        sample_explanation = self._explain_single_sample(
            shap_values_array[sample_idx],
            X_continuous[sample_idx],
            feature_names,
            predictions[sample_idx]
        )
        
        logger.info(f"\n📋 DETAILED EXPLANATION FOR SAMPLE {sample_idx}:")
        logger.info(f"   Predicted Energy: {predictions[sample_idx]:.2f}W")
        
        # Get base value
        try:
            base_val = float(self.shap_explainer.expected_value)
            logger.info(f"   Base Value: {base_val:.2f}W")
        except:
            base_val = predictions.mean()
            logger.info(f"   Base Value (estimated): {base_val:.2f}W")
        
        logger.info(f"   Top positive contributors:")
        for feat, val in sample_explanation['top_positive'][:5]:
            logger.info(f"      + {feat}: {val:+.3f}")
        logger.info(f"   Top negative contributors:")
        for feat, val in sample_explanation['top_negative'][:5]:
            logger.info(f"      - {feat}: {val:+.3f}")
        
        # Generate plots
        if plot:
            plot_path = save_path or 'shap_explanation.png'
            self._plot_shap_explanation(
                shap_values_array, 
                X_continuous, 
                feature_names,
                predictions,
                sample_idx,
                plot_path
            )
        
        # Compile result
        result = {
            'shap_values': shap_values_array,
            'feature_names': feature_names,
            'feature_importance': importance_df,
            'base_value': base_val,
            'predictions': predictions,
            'sample_explanation': sample_explanation
        }
        
        logger.info("=" * 80)
        logger.info("✓ SHAP EXPLANATION COMPLETE")
        logger.info("=" * 80)
        
        return result
    
    # Keep _explain_single_sample as is
    
    # Add this method after the explain_prediction method (around line 440):

    def _explain_single_sample(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: List[str],
        prediction: float
    ) -> Dict[str, Any]:
        """
        Create detailed explanation for a single sample
        
        Args:
            shap_values: SHAP values for the sample
            feature_values: Actual feature values
            feature_names: Names of features
            prediction: Model prediction
        
        Returns:
            Dictionary with explanation details
        """
        # Pair features with SHAP values
        feature_contributions = list(zip(feature_names, shap_values, feature_values))
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Separate positive and negative contributions
        positive = [(f, s) for f, s, v in feature_contributions if s > 0]
        negative = [(f, s) for f, s, v in feature_contributions if s < 0]
        
        return {
            'prediction': prediction,
            'top_positive': positive,
            'top_negative': negative,
            'all_contributions': feature_contributions
        }

    # Replace the _plot_shap_explanation method (around line 489):

    def _plot_shap_explanation(
        self, 
        shap_values: np.ndarray, 
        X: np.ndarray,
        feature_names: List[str],
        predictions: np.ndarray,
        sample_idx: int,
        save_path: str
    ):
        """
        Generate sophisticated SHAP visualization plots (saved separately)
        """
        logger.info(f"Generating SHAP plots...")
        
        # Get base value
        try:
            base_val = float(self.shap_explainer.expected_value)
        except:
            base_val = predictions.mean()
        
        # Directory for saving plots
        save_dir = Path(save_path).parent
        save_stem = Path(save_path).stem
        
        # ============================================================
        # PLOT 1: Feature Importance Bar Chart (Enhanced)
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True).tail(20)  # Top 20
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                       color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(val, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', 
                   va='center', ha='left', fontsize=10, 
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
        ax.set_title('Top 20 Most Important Features\n(Global Feature Importance)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plot1_path = save_dir / f"{save_stem}_1_importance_bar.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"  ✓ Plot 1 saved: {plot1_path}")
        plt.close()
        
        # ============================================================
        # PLOT 2: Beeswarm Plot (Feature Impact Distribution)
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.sca(ax)
        
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=feature_names,
            max_display=20,
            show=False,
            cmap='coolwarm',
            plot_size=(12, 10)
        )
        
        ax.set_title('Feature Impact Distribution (Beeswarm Plot)\n'
                    'Color: Feature Value (Red=High, Blue=Low)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x', linestyle='--')
        
        plt.tight_layout()
        plot2_path = save_dir / f"{save_stem}_2_beeswarm.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"  ✓ Plot 2 saved: {plot2_path}")
        plt.close()
        
        # ============================================================
        # PLOT 3: Waterfall Plot for Single Sample
        # ============================================================
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.sca(ax)
        
        try:
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=base_val,
                    data=X[sample_idx],
                    feature_names=feature_names
                ),
                max_display=20,
                show=False
            )
            ax.set_title(f'Waterfall Explanation - Sample {sample_idx}\n'
                        f'Prediction: {predictions[sample_idx]:.2f}W | Base Value: {base_val:.2f}W',
                        fontsize=16, fontweight='bold', pad=20)
        except Exception as e:
            logger.warning(f"Waterfall plot failed: {e}, using bar chart fallback")
            
            # Fallback: Manual bar chart
            top_n = 20
            top_features_idx = np.argsort(np.abs(shap_values[sample_idx]))[-top_n:][::-1]
            values = shap_values[sample_idx][top_features_idx]
            names = [feature_names[i] for i in top_features_idx]
            
            colors_bar = ['#d62728' if v < 0 else '#2ca02c' for v in values]
            bars = ax.barh(names, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bar, val in zip(bars, values):
                label_x = val + (0.05 if val > 0 else -0.05)
                ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', 
                       va='center', ha='left' if val > 0 else 'right',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
            ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
            ax.set_title(f'Feature Contributions - Sample {sample_idx}\n'
                        f'Prediction: {predictions[sample_idx]:.2f}W | Base: {base_val:.2f}W',
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        plot3_path = save_dir / f"{save_stem}_3_waterfall_sample{sample_idx}.png"
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"  ✓ Plot 3 saved: {plot3_path}")
        plt.close()
        
        # ============================================================
        # PLOT 4: Top 10 Positive & Negative Contributors
        # ============================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get top contributors
        sorted_idx = np.argsort(shap_values[sample_idx])
        top_negative = sorted_idx[:10]
        top_positive = sorted_idx[-10:][::-1]
        
        # Negative contributors (left panel)
        neg_values = shap_values[sample_idx][top_negative]
        neg_names = [feature_names[i] for i in top_negative]
        
        bars1 = ax1.barh(neg_names, neg_values, color='#d62728', alpha=0.8, 
                         edgecolor='darkred', linewidth=1.5)
        
        for bar, val in zip(bars1, neg_values):
            ax1.text(val - 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', 
                    va='center', ha='right', fontsize=10, fontweight='bold',
                    color='white')
        
        ax1.axvline(0, color='black', linewidth=2)
        ax1.set_xlabel('SHAP Value (Negative Impact)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Features DECREASING Energy', fontsize=14, fontweight='bold', color='darkred')
        ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Positive contributors (right panel)
        pos_values = shap_values[sample_idx][top_positive]
        pos_names = [feature_names[i] for i in top_positive]
        
        bars2 = ax2.barh(pos_names, pos_values, color='#2ca02c', alpha=0.8, 
                         edgecolor='darkgreen', linewidth=1.5)
        
        for bar, val in zip(bars2, pos_values):
            ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', 
                    va='center', ha='left', fontsize=10, fontweight='bold',
                    color='white')
        
        ax2.axvline(0, color='black', linewidth=2)
        ax2.set_xlabel('SHAP Value (Positive Impact)', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Features INCREASING Energy', fontsize=14, fontweight='bold', color='darkgreen')
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.suptitle(f'Feature Contributions Breakdown - Sample {sample_idx}\n'
                    f'Final Prediction: {predictions[sample_idx]:.2f}W',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plot4_path = save_dir / f"{save_stem}_4_pos_neg_contributors.png"
        plt.savefig(plot4_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"  ✓ Plot 4 saved: {plot4_path}")
        plt.close()
        
        # ============================================================
        # PLOT 5: Heatmap of SHAP values (all samples)
        # ============================================================
        if len(shap_values) > 1:
            fig, ax = plt.subplots(figsize=(14, max(6, len(shap_values)*0.3)))
            
            # Select top 15 features by importance
            top_features_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-15:][::-1]
            shap_subset = shap_values[:, top_features_idx]
            feature_subset = [feature_names[i] for i in top_features_idx]
            
            im = ax.imshow(shap_subset.T, cmap='RdBu_r', aspect='auto', 
                          vmin=-np.abs(shap_subset).max(), 
                          vmax=np.abs(shap_subset).max())
            
            ax.set_yticks(range(len(feature_subset)))
            ax.set_yticklabels(feature_subset, fontsize=11)
            ax.set_xticks(range(len(shap_values)))
            ax.set_xticklabels([f'S{i}' for i in range(len(shap_values))], fontsize=10)
            ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
            ax.set_title('SHAP Values Heatmap (Top 15 Features)\n'
                        'Red=Increases Energy | Blue=Decreases Energy',
                        fontsize=16, fontweight='bold', pad=20)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('SHAP Value', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plot5_path = save_dir / f"{save_stem}_5_heatmap.png"
            plt.savefig(plot5_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"  ✓ Plot 5 saved: {plot5_path}")
            plt.close()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ ALL SHAP PLOTS GENERATED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"Plots saved in: {save_dir}")
    
    def get_feature_importance(
        self,
        n_samples: int = 3,  # Reduced from 1000
        save_json: bool = True
    ) -> pd.DataFrame:
        """
        Get global feature importance based on SHAP values
        
        Args:
            n_samples: Number of samples to use (KernelExplainer is slow, use small number)
            save_json: Whether to save importance to JSON file
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.enable_shap:
            logger.warning("SHAP is disabled. Enable with enable_shap=True")
            return pd.DataFrame()
        
        if self.shap_explainer is None:
            self.initialize_shap()
        
        logger.info(f"Calculating global feature importance using {n_samples} samples...")
        logger.info("NOTE: This may take 5-10 minutes with KernelExplainer...")
        
        # Load data
        train_data = pd.read_csv("../archive/merged_training_data.csv", nrows=n_samples)
        result = self.explain_prediction(train_data, plot=True, n_samples=n_samples)
        
        importance_df = result.get('feature_importance', pd.DataFrame())
        
        if not importance_df.empty and save_json:
            importance_dict = importance_df.to_dict('records')
            with open('feature_importance.json', 'w') as f:
                json.dump(importance_dict, f, indent=2)
            logger.info("✓ Feature importance saved to: feature_importance.json")
        
        return importance_df


# ==================== CLI Interface ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Keras Energy Prediction Inference with SHAP")
    parser.add_argument('--csv', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate if ground truth exists')
    parser.add_argument('--shap', action='store_true', help='Enable SHAP explanations')
    parser.add_argument('--shap-samples', type=int, default=3, help='Number of samples for SHAP')
    parser.add_argument('--importance', action='store_true', help='Calculate global feature importance')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = KerasEnergyPredictor(enable_shap=args.shap)
    
    if args.csv:
        # Batch prediction from CSV
        df_pred = predictor.predict_batch_from_csv(
            csv_path=args.csv,
            output_path=args.output
        )
        
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS (first 5 rows)")
        print("=" * 80)
        cols = ['Time', 'BS', 'Energy_Predicted']
        if 'Energy' in df_pred.columns:
            cols.insert(2, 'Energy')
            cols.append('Error')
        print(df_pred[cols].head())
        
        # SHAP explanation if requested
        if args.shap:
            df_sample = df_pred.head(args.shap_samples)
            predictor.explain_prediction(
                df_sample, 
                sample_idx=0,
                plot=True,
                save_path='shap_batch_explanation.png'
            )
    
    elif args.importance:
        # Calculate feature importance
        importance_df = predictor.get_feature_importance(n_samples=args.shap_samples)
        print("\n" + "=" * 80)
        print("GLOBAL FEATURE IMPORTANCE")
        print("=" * 80)
        print(importance_df.head(20))
        
    else:
        # Example: Predict on sample from merged training data
        print("\n" + "=" * 80)
        print("EXAMPLE: Prediction on Sample Data")
        print("=" * 80)
        
        # Load one sample from training data
        sample_csv = "../archive/merged_training_data.csv"
        df_sample = pd.read_csv(sample_csv, nrows=10)
        
        print("\nSample input data (first 3 rows):")
        print(df_sample[['Time', 'BS', 'load', 'RUType', 'Mode']].head(3))
        
        # Make predictions
        predictions = predictor.predict(df_sample)
        
        print("\n" + "=" * 80)
        print("PREDICTIONS:")
        print("=" * 80)
        for i, (idx, row) in enumerate(df_sample.iterrows()):
            actual = row.get('Energy', None)
            pred = predictions[i]
            
            if actual is not None:
                error = abs(actual - pred)
                print(f"{i+1}. BS={row['BS']}, Time={row['Time']}")
                print(f"   Actual: {actual:.2f}W, Predicted: {pred:.2f}W, Error: {error:.2f}W")
            else:
                print(f"{i+1}. BS={row['BS']}, Time={row['Time']}")
                print(f"   Predicted: {pred:.2f}W")
        
        print("=" * 80)
        
        # SHAP explanation if requested
        if args.shap:
            print("\n" + "=" * 80)
            print("GENERATING SHAP EXPLANATIONS")
            print("=" * 80)
            predictor.explain_prediction(
                df_sample,
                sample_idx=0,
                plot=True,
                save_path='shap_example_explanation.png'
            )