import torch
import torch.nn as nn
import argparse
from lib.BaseModelTrainer import BaseModelTrainer
from config import FNN_CONFIG

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super(FeedforwardNN, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
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
            dropout_rate=self.config['dropout_rate']
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FNN for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    trainer = FNNTrainer()
    trainer.train(args.data_path) 