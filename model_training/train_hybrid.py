import torch
import torch.nn as nn
import argparse
from lib.BaseModelTrainer import BaseModelTrainer
from config import HYBRID_CONFIG

class HybridController(nn.Module):
    def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers,
                 output_size, dropout_rate, sequence_length):
        super(HybridController, self).__init__()
        
        # Reshape input for CNN
        self.sequence_length = sequence_length
        
        # CNN layers for spatial feature extraction
        cnn_layers = []
        in_channels = 1  # Single channel input
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        cnn_output_size = (input_size // (2 ** len(cnn_channels))) * cnn_channels[-1]
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        
        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch, sequence, features) -> (batch * sequence, 1, features)
        x = x.view(-1, 1, x.size(-1))
        
        # Apply CNN
        x = self.cnn(x)
        
        # Reshape for LSTM: (batch * sequence, channels, features) -> (batch, sequence, features)
        x = x.view(batch_size, self.sequence_length, -1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output
        x = lstm_out[:, -1, :]
        
        # Final fully connected layers
        return self.fc(x)

class HybridTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__('hybrid', HYBRID_CONFIG)
    
    def create_model(self):
        return HybridController(
            input_size=self.config['input_size'],
            cnn_channels=self.config['cnn_channels'],
            lstm_hidden_size=self.config['lstm_hidden_size'],
            lstm_num_layers=self.config['lstm_num_layers'],
            output_size=self.config['output_size'],
            dropout_rate=self.config['dropout_rate'],
            sequence_length=self.config['sequence_length']
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-LSTM for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    trainer = HybridTrainer()
    trainer.train(args.data_path) 