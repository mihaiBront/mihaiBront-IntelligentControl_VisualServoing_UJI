import torch
import torch.nn as nn
import argparse
from lib.BaseModelTrainer import BaseModelTrainer
from config import RESNET_CONFIG

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

class ResNetController(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, output_size, dropout_rate):
        super(ResNetController, self).__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        return self.output_proj(x)

class ResNetTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__('resnet', RESNET_CONFIG)
    
    def create_model(self):
        return ResNetController(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_blocks=self.config['num_blocks'],
            output_size=self.config['output_size'],
            dropout_rate=self.config['dropout_rate']
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    trainer = ResNetTrainer()
    trainer.train(args.data_path) 