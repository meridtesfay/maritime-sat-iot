"""
Hybrid CNN-LSTM Model with Attention Fusion
Combines spatial CNN and temporal LSTM branches
"""

import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_size=15, sequence_length=24, num_classes=3, dropout=0.3):
        super().__init__()
        
        # CNN branch for spatial features
        self.cnn_conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.cnn_bn1 = nn.BatchNorm1d(64)
        self.cnn_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.cnn_bn2 = nn.BatchNorm1d(128)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM branch for temporal features
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Multi-head attention fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Linear(128 + 256, 512)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)
        
        # CNN branch
        x_cnn = x.transpose(1, 2)  # (batch, features, seq_len)
        x_cnn = self.relu(self.cnn_bn1(self.cnn_conv1(x_cnn)))
        x_cnn = self.relu(self.cnn_bn2(self.cnn_conv2(x_cnn)))
        x_cnn = self.cnn_pool(x_cnn).squeeze(-1)  # (batch, 128)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)  # (batch, seq_len, 256)
        
        # Attention over LSTM outputs
        x_lstm_attn, _ = self.attention(x_lstm, x_lstm, x_lstm)
        x_lstm = x_lstm_attn.mean(dim=1)  # (batch, 256)
        
        # Fusion
        x_fused = torch.cat([x_cnn, x_lstm], dim=1)  # (batch, 384)
        x_fused = self.relu(self.fusion(x_fused))
        
        # Classifier
        x = self.dropout(x_fused)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = HybridModel()
    x = torch.randn(32, 24, 15)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
