"""
Keras Model Architecture for Energy Prediction
Replicates PyTorch EnergyPredictionANN with Functional API and separate categorical embeddings
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import GlorotUniform, Constant
import numpy as np

class Swish(layers.Layer):
    """Custom Swish activation: x * sigmoid(beta * x)"""
    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = beta
    
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(self.beta * inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"beta": self.beta})
        return config

def build_energy_prediction_model(
    continuous_dim: int = 46,
    categorical_dims: dict = None,
    hidden_dims: list = None,
    dropout_rate: float = 0.2,
    use_batch_norm: bool = True
) -> Model:
    """
    Build Keras Functional API model with separate categorical embeddings
    
    Args:
        continuous_dim: Number of continuous features
        categorical_dims: Dict mapping feature name to vocabulary size
            e.g., {'RUType': 12, 'Mode': 2, 'load_bin': 101}
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        use_batch_norm: Whether to use batch normalization
    
    Returns:
        Compiled Keras Model
    """
    if categorical_dims is None:
        categorical_dims = {'RUType': 12, 'Mode': 2, 'load_bin': 101}
    
    if hidden_dims is None:
        hidden_dims = [128, 256, 128, 64]
    
    # Calculate embedding dimensions (same logic as PyTorch)
    embedding_dims = {
        name: min(50, max(3, vocab_size // 2))
        for name, vocab_size in categorical_dims.items()
    }
    
    # ==================== INPUT LAYERS ====================
    # Continuous input
    continuous_input = layers.Input(
        shape=(continuous_dim,), 
        name='continuous',
        dtype=tf.float32
    )
    
    # Categorical inputs (one per feature)
    categorical_inputs = {}
    embedded_categoricals = []
    
    for name, vocab_size in categorical_dims.items():
        # Input layer for this categorical feature
        cat_input = layers.Input(
            shape=(1,), 
            name=name,
            dtype=tf.int32
        )
        categorical_inputs[name] = cat_input
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dims[name],
            embeddings_initializer=GlorotUniform(seed=42),
            name=f'{name}_embedding'
        )(cat_input)
        
        # Flatten embedding (from (batch, 1, emb_dim) to (batch, emb_dim))
        embedding_flat = layers.Flatten(name=f'{name}_flatten')(embedding)
        embedded_categoricals.append(embedding_flat)
    
    # ==================== CONCATENATE ALL FEATURES ====================
    all_features = layers.Concatenate(name='concat_features')(
        [continuous_input] + embedded_categoricals
    )
    
    # ==================== HIDDEN LAYERS ====================
    x = all_features
    
    for i, hidden_dim in enumerate(hidden_dims):
        # Dense layer
        x = layers.Dense(
            hidden_dim,
            kernel_initializer=GlorotUniform(seed=42),
            bias_initializer=Constant(0.0),
            name=f'dense_{i+1}'
        )(x)
        
        # Swish activation
        x = Swish(beta=1.0, name=f'swish_{i+1}')(x)
        
        # Batch normalization
        if use_batch_norm:
            x = layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5,
                name=f'batch_norm_{i+1}'
            )(x)
        
        # Dropout
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # ==================== OUTPUT LAYER ====================
    output = layers.Dense(
        1,
        activation='linear',
        kernel_initializer=GlorotUniform(seed=42),
        bias_initializer=Constant(0.0),
        name='output'
    )(x)
    
    # ==================== BUILD MODEL ====================
    model = Model(
        inputs=[continuous_input] + list(categorical_inputs.values()),
        outputs=output,
        name='EnergyPredictionModel'
    )
    
    return model

def compile_model(model: Model, learning_rate: float = 0.001) -> Model:
    """
    Compile model with optimizer and loss
    
    Args:
        model: Keras model
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=[
            'mae',
            keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    
    return model

def get_model_summary(model: Model) -> str:
    """Get detailed model summary as string"""
    from io import StringIO
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()

if __name__ == "__main__":
    # Test model creation
    print("Building Keras Energy Prediction Model...")
    print("=" * 80)
    
    model = build_energy_prediction_model(
        continuous_dim=46,
        categorical_dims={'RUType': 12, 'Mode': 2, 'load_bin': 101},
        hidden_dims=[128, 256, 128, 64],
        dropout_rate=0.2
    )
    
    model = compile_model(model, learning_rate=0.001)
    
    print(get_model_summary(model))
    
    # Test with dummy data
    print("\nTesting model with dummy data...")
    batch_size = 4
    dummy_continuous = np.random.randn(batch_size, 46).astype(np.float32)
    dummy_rutype = np.random.randint(0, 12, (batch_size, 1)).astype(np.int32)
    dummy_mode = np.random.randint(0, 2, (batch_size, 1)).astype(np.int32)
    dummy_load_bin = np.random.randint(0, 101, (batch_size, 1)).astype(np.int32)
    
    predictions = model.predict({
        'continuous': dummy_continuous,
        'RUType': dummy_rutype,
        'Mode': dummy_mode,
        'load_bin': dummy_load_bin
    }, verbose=0)
    
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Sample predictions: {predictions[:3].flatten()}")
    print("\n✓ Model architecture created successfully!")