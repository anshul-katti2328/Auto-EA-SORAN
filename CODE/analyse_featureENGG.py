import sys
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import savgol_filter
from sklearn.preprocessing import SplineTransformer

# Define a minimal FarziFeatureEngineer class to enable unpickling
class FarziFeatureEngineer:
    """Minimal class definition to load pickled feature engineer"""
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.categorical_features = []
        self.feature_names = []
        self.is_fitted = False
        self.hour_spline_transformer = None
        self.load_spline_transformer = None

# Now load the pickle
with open('models/feature_engineer.pkl', 'rb') as f:
    fe = pickle.load(f)

print("=" * 80)
print("FEATURE ENGINEER ANALYSIS FOR KERAS MIGRATION")
print("=" * 80)

print("\n1. CATEGORICAL FEATURES ANALYSIS")
print("-" * 80)
if hasattr(fe, 'label_encoders') and fe.label_encoders:
    total_categorical = len(fe.label_encoders)
    print(f"Total categorical features: {total_categorical}\n")
    
    for name, encoder in fe.label_encoders.items():
        vocab_size = len(encoder.classes_)
        suggested_emb_dim = min(50, max(3, vocab_size // 2))
        
        print(f"Feature: {name}")
        print(f"  ├─ Vocabulary size: {vocab_size}")
        print(f"  ├─ Unique classes: {list(encoder.classes_)}")
        print(f"  └─ Suggested embedding dimension: {suggested_emb_dim}")
        print()
else:
    print("  ⚠️  No label encoders found!")

print("\n2. CONTINUOUS FEATURES ANALYSIS")
print("-" * 80)
if hasattr(fe, 'feature_names') and fe.feature_names:
    categorical_features = getattr(fe, 'categorical_features', [])
    continuous_features = [f for f in fe.feature_names if f not in categorical_features]
    
    print(f"Total features: {len(fe.feature_names)}")
    print(f"  ├─ Continuous: {len(continuous_features)}")
    print(f"  └─ Categorical: {len(categorical_features)}")
    
    print(f"\nContinuous feature names (first 20):")
    for i, feat in enumerate(continuous_features[:20], 1):
        print(f"  {i:2d}. {feat}")
    
    if len(continuous_features) > 20:
        print(f"  ... and {len(continuous_features) - 20} more features")
else:
    print("  ⚠️  No feature names found!")

print("\n3. SCALER INFORMATION")
print("-" * 80)
if hasattr(fe, 'scaler') and fe.scaler:
    print(f"Scaler type: {type(fe.scaler).__name__}")
    if hasattr(fe.scaler, 'n_features_in_'):
        print(f"Features scaled: {fe.scaler.n_features_in_}")
    if hasattr(fe.scaler, 'mean_'):
        print(f"Mean vector shape: {fe.scaler.mean_.shape}")
        print(f"Scale vector shape: {fe.scaler.scale_.shape}")
        print(f"Mean range: [{fe.scaler.mean_.min():.4f}, {fe.scaler.mean_.max():.4f}]")
        print(f"Scale range: [{fe.scaler.scale_.min():.4f}, {fe.scaler.scale_.max():.4f}]")
else:
    print("  ⚠️  No scaler found!")

print("\n4. OTHER ATTRIBUTES")
print("-" * 80)
print(f"Is fitted: {getattr(fe, 'is_fitted', 'Unknown')}")
print(f"Has hour spline transformer: {hasattr(fe, 'hour_spline_transformer')}")
print(f"Has load spline transformer: {hasattr(fe, 'load_spline_transformer')}")

print("\n" + "=" * 80)
print("KERAS MODEL CONFIGURATION SUMMARY")
print("=" * 80)

if hasattr(fe, 'feature_names') and hasattr(fe, 'label_encoders'):
    categorical_features = getattr(fe, 'categorical_features', [])
    continuous_features = [f for f in fe.feature_names if f not in categorical_features]
    
    print(f"\n✓ Input Layer Configuration:")
    print(f"  ├─ Continuous Input: shape=({len(continuous_features)},), dtype=float32")
    print(f"  └─ Categorical Inputs ({len(fe.label_encoders)} total):")
    
    total_embedding_params = 0
    for i, (name, encoder) in enumerate(fe.label_encoders.items(), 1):
        vocab_size = len(encoder.classes_)
        suggested_emb_dim = min(50, max(3, vocab_size // 2))
        embedding_params = vocab_size * suggested_emb_dim
        total_embedding_params += embedding_params
        
        is_last = (i == len(fe.label_encoders))
        prefix = "      └─" if is_last else "      ├─"
        
        print(f"{prefix} {name}:")
        print(f"      {'  ' if not is_last else '   '}  ├─ shape: (1,), dtype=int32")
        print(f"      {'  ' if not is_last else '   '}  ├─ vocab_size: {vocab_size}")
        print(f"      {'  ' if not is_last else '   '}  ├─ embedding_dim: {suggested_emb_dim}")
        print(f"      {'  ' if not is_last else '   '}  └─ trainable_params: {embedding_params:,}")
    
    print(f"\n✓ Model Architecture:")
    total_embedded = len(continuous_features) + sum(min(50, max(3, len(e.classes_) // 2)) for e in fe.label_encoders.values())
    print(f"  ├─ Total input features after embedding: {total_embedded}")
    print(f"  ├─ Embedding parameters: {total_embedding_params:,}")
    print(f"  └─ Suggested hidden layers: [128, 256, 128, 64] with Swish activation")

print("\n" + "=" * 80)
print("TRAINING DATA LOCATION")
print("=" * 80)
print("✓ Training CSVs found in: ../archive/")
print("  ├─ imgs_2023071012133740345.csv (train)")
print("  ├─ imgs_202307101549519358.csv (test)")
print("  ├─ imgs_2023071012130978799.csv (cell)")
print("  └─ imgs_2023071012123392536.csv (BS)")
print("\n✓ Model save location: models/")
print("  ├─ Current: energy_prediction_model.pth (PyTorch)")
print("  └─ New: energy_prediction_keras.keras (to be created)")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Install required packages:")
print("   pip install tensorflow keras")
print("\n2. I will create the following files:")
print("   ├─ keras_model_architecture.py")
print("   ├─ keras_trainer.py")
print("   ├─ keras_inference_engine.py")
print("   ├─ keras_data_utils.py")
print("   └─ Updated newPredictionAgent.py")
print("\n3. Train the Keras model:")
print("   python keras_trainer.py")
print("\n4. Test inference:")
print("   python test_keras_inference.py")
print("\n" + "=" * 80)