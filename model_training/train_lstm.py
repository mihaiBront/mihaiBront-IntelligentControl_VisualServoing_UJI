import torch
import torch.nn as nn
import argparse
from lib.BaseModelTrainer import BaseModelTrainer
from config import LSTM_CONFIG

class LSTMController(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMController, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class LSTMTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__('lstm', LSTM_CONFIG)
    
    def create_model(self):
        return LSTMController(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout_rate=self.config['dropout_rate']
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM for IBVS control')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    trainer = LSTMTrainer()
    trainer.train(args.data_path) 