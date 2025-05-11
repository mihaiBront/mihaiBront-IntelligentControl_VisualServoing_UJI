"""
Configuration file for model training
"""

# Common configurations
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
DEVICE = 'cuda'  # or 'cpu' if GPU not available

# FNN configurations
FNN_CONFIG = {
    'input_size': 8,  # 8 feature coordinates
    'hidden_sizes': [128, 64],
    'output_size': 6,  # 6 DOF velocity commands
    'dropout_rate': 0.2
}

# LSTM configurations
LSTM_CONFIG = {
    'input_size': 8,
    'hidden_size': 64,
    'num_layers': 2,
    'output_size': 6,
    'dropout_rate': 0.2,
    'sequence_length': 10
}

# ResNet configurations
RESNET_CONFIG = {
    'input_size': 8,
    'hidden_size': 64,
    'num_blocks': 3,
    'output_size': 6,
    'dropout_rate': 0.2
}

# Hybrid configurations
HYBRID_CONFIG = {
    'input_size': 8,
    'cnn_channels': [16, 32],
    'lstm_hidden_size': 64,
    'lstm_num_layers': 2,
    'output_size': 6,
    'dropout_rate': 0.2,
    'sequence_length': 10
} 