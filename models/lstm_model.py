"""
BiLSTM Model - Bidirectional LSTM for temporal dynamics
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, 
                 num_classes=3, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classifier
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

if __name__ == "__main__":
    model = LSTMModel()
    x = torch.randn(32, 24, 15)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
