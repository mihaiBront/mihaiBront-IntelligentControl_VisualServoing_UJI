import torch
import torch.nn as nn
import argparse
from lib.BaseModelTrainer import BaseModelTrainer
from config import FNN_CONFIG

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate, use_batch_norm=True):
        super(FeedforwardNN, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            linear = nn.Linear(prev_size, hidden_size)
            # Initialize weights using He initialization for ReLU
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            
            # Batch normalization (before activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation and dropout
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer with careful initialization
        output_layer = nn.Linear(prev_size, output_size)
        # Initialize output layer with smaller weights for stability
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.model = nn.Sequential(*layers)
        
        # Apply weight initialization to all layers
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Skip if already initialized (output layer)
            if not hasattr(module, '_initialized'):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                module._initialized = True
    
    def forward(self, x):
        return self.model(x)

class FNNTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__('fnn', FNN_CONFIG)
    
    def create_model(self):
        return FeedforwardNN(
            input_size=self.config['input_size'],
            hidden_sizes=self.config['hidden_sizes'],
            output_size=self.config['output_size'],
            dropout_rate=self.config['dropout_rate'],
            use_batch_norm=self.config.get('use_batch_norm', True)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FNN for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    trainer = FNNTrainer()
    trainer.train(args.data_path) 