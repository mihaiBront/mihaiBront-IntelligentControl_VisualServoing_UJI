"""
Configuration file for model training
"""

# Common configurations
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 150
LEARNING_RATE = 0.0005  # Reduced learning rate for more stable training
VAL_SPLIT = 0.2
DEVICE = 'cuda'  # or 'cpu' if GPU not available

# Early stopping configurations
EARLY_STOPPING_PATIENCE = 15  # Reduced patience to stop earlier when overfitting
EARLY_STOPPING_MIN_DELTA = 1e-5  # Slightly larger threshold

# FNN configurations
FNN_CONFIG = {
    'input_size': 8,  # 8 feature coordinates
    'hidden_sizes': [64, 32],  # Even simpler architecture to reduce overfitting
    'output_size': 3,  # 3 DOF linear velocity commands (vx, vy, vz only)
    'dropout_rate': 0.3,  # Increased dropout for more regularization
    'use_batch_norm': True  # Added batch normalization
}

# LSTM configurations
LSTM_CONFIG = {
    'input_size': 8,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 3,  # 3 DOF linear velocity commands
    'dropout_rate': 0.2,
    'sequence_length': 10
}

# ResNet configurations
RESNET_CONFIG = {
    'input_size': 8,
    'hidden_size': 64,
    'num_blocks': 3,
    'output_size': 3,  # 3 DOF linear velocity commands
    'dropout_rate': 0.2
}

# Hybrid configurations
HYBRID_CONFIG = {
    'input_size': 8,
    'cnn_channels': [16, 32],
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'output_size': 3,  # 3 DOF linear velocity commands
    'dropout_rate': 0.2,
    'sequence_length': 10
} 