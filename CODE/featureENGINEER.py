"""
Shared FarziFeatureEngineer class
Used by both PyTorch (old) and Keras (new) pipelines
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, SplineTransformer
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)

class FarziFeatureEngineer:
    """
    Feature engineering pipeline for 5G energy prediction
    Compatible with both PyTorch and Keras models
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_features = ['RUType', 'Mode', 'load_bin']
        self.feature_names = []
        self.continuous_feature_names = []
        self.is_fitted = False
        self.hour_spline_transformer = None
        self.load_spline_transformer = None

    # Find the apply_farzi_preprocessing method and update the smoothed load section (around line 95):

    def apply_farzi_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Farzi's preprocessing steps"""
        df = df.copy()
        
        # Handle different column name variations
        if 'BS' in df.columns and 'BSID' not in df.columns:
            df['BSID'] = df['BS']
        elif 'BSID' not in df.columns:
            df['BSID'] = 'BS_0'
        
        if 'CellName' in df.columns and 'CellID' not in df.columns:
            df['CellID'] = df['CellName']
        elif 'CellID' not in df.columns:
            df['CellID'] = 'Cell_0'
        
        # Time features
        df['Time'] = pd.to_datetime(df['Time'])
        df['day'] = df['Time'].dt.day
        df['weekday_number'] = df['Time'].dt.dayofweek
        df['hour'] = df['Time'].dt.hour
        
        # Hour splines (periodic)
        if not self.is_fitted:
            self.hour_spline_transformer = SplineTransformer(
                n_knots=12, degree=3, include_bias=False
            )
            hour_splines = self.hour_spline_transformer.fit_transform(
                df['hour'].values.reshape(-1, 1)
            )
        else:
            hour_splines = self.hour_spline_transformer.transform(
                df['hour'].values.reshape(-1, 1)
            )
        
        for i in range(hour_splines.shape[1]):
            df[f'hour_spline_{i}'] = hour_splines[:, i]
        
        # Add w feature if missing
        if 'w' not in df.columns:
            df['w'] = 1.0
        
        # Load binning
        if "load" in df.columns:
            load_values = (
                pd.to_numeric(
                df["load"],
                errors="coerce",
                )
                .fillna(0.0)
                .clip(0.0, 1.0)
            )
            df["load_bin"] = np.minimum(
                np.floor(load_values * 100.0).astype(int),
                99,
            )
        else:
            df["load_bin"] = 0
        
        # Load splines
        if 'load' in df.columns and not df['load'].isna().all():
            if not self.is_fitted:
                self.load_spline_transformer = SplineTransformer(
                    n_knots=10, degree=3, include_bias=False
                )
                load_splines = self.load_spline_transformer.fit_transform(
                    df['load'].values.reshape(-1, 1)
                )
            else:
                load_splines = self.load_spline_transformer.transform(
                    df['load'].values.reshape(-1, 1)
                )
            
            for i in range(load_splines.shape[1]):
                df[f'load_spline_{i}'] = load_splines[:, i]
        
        # Lag features
        if 'load' in df.columns:
            df['load_lag_1'] = df.groupby('BSID')['load'].shift(1)
            df['load_lag_2'] = df.groupby('BSID')['load'].shift(2)
            df.fillna({
                'load_lag_1': df['load'].mean(), 
                'load_lag_2': df['load'].mean()
            }, inplace=True)
        
        # Smoothed load - UPDATED SECTION
        if 'load' in df.columns:
            # Check if we have enough data for smoothing
            min_window = 5
            if len(df) >= min_window:
                try:
                    df['load_smoothed'] = savgol_filter(
                        df['load'], 
                        window_length=min_window, 
                        polyorder=2
                    )
                except:
                    # If smoothing fails, use original load
                    df['load_smoothed'] = df['load']
            else:
                # Not enough data, use original load
                df['load_smoothed'] = df['load']
        else:
            # No load column, create dummy
            df['load_smoothed'] = 0.0
        
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Energy'
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Fit and transform data"""
        logger.info("Applying preprocessing...")
        df_processed = self.apply_farzi_preprocessing(df)
        
        # Auto-detect categorical features
        potential_categoricals = ['RUType', 'Mode', 'load_bin', 'RU_Type', 'mode']
        self.categorical_features = [
            col for col in potential_categoricals 
            if col in df_processed.columns
        ]
        
        logger.info(f"Found categorical features: {self.categorical_features}")
        
        # Identify continuous features (only numeric)
        exclude_cols = [
            'Time', 'BSID', 'CellID', target_col, 'Unnamed: 0', 'BS', 'CellName'
        ] + self.categorical_features
        
        all_cols = df_processed.columns.tolist()
        continuous_features = []
        
        for col in all_cols:
            if col in exclude_cols:
                continue
            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                continuous_features.append(col)
            else:
                logger.warning(f"Skipping non-numeric column: {col}")
        
        self.continuous_feature_names = continuous_features
        logger.info(f"Identified {len(continuous_features)} continuous features")
        
        # Encode categoricals
        X_categorical = {}
        for cat_feat in self.categorical_features:
            if cat_feat in df_processed.columns:
                self.label_encoders[cat_feat] = LabelEncoder()
                X_categorical[cat_feat] = self.label_encoders[cat_feat].fit_transform(
                    df_processed[cat_feat].astype(str)
                )
                logger.info(f"Encoded {cat_feat}: {len(self.label_encoders[cat_feat].classes_)} classes")
        
        # Scale continuous
        X_continuous = self.scaler.fit_transform(df_processed[continuous_features])
        
        # Target
        y = df_processed[target_col].values if target_col in df_processed.columns else None
        
        self.feature_names = continuous_features + self.categorical_features
        self.is_fitted = True
        
        logger.info(f"✓ {len(continuous_features)} continuous, {len(X_categorical)} categorical")
        
        return X_continuous, X_categorical, y

    def transform(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Transform without fitting"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        
        df_processed = self.apply_farzi_preprocessing(df)
        
        # Use stored continuous feature names
        continuous_features = self.continuous_feature_names
        
        # Encode categoricals
        X_categorical = {}
        for cat_feat in self.categorical_features:
            if cat_feat in df_processed.columns:
                X_categorical[cat_feat] = self.label_encoders[cat_feat].transform(
                    df_processed[cat_feat].astype(str)
                )
        
        # Scale continuous
        X_continuous = self.scaler.transform(df_processed[continuous_features])
        
        return X_continuous, X_categorical