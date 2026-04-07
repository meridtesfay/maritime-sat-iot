"""
Transformer Model - Multi-head self-attention for long-range dependencies
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_size=15, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, num_classes=3, dropout=0.3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classifier
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = TransformerModel()
    x = torch.randn(32, 24, 15)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
