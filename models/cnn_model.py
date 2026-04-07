"""
CNN Model - 1D ResNet for spatial feature extraction
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size=15, sequence_length=24, num_classes=3, dropout=0.3):
        super().__init__()
        
        # Input: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
        
        # Initial conv
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Residual blocks
        self.layer1 = ResidualBlock(64, 128)
        self.layer2 = ResidualBlock(128, 256)
        self.layer3 = ResidualBlock(256, 512)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # -> (batch, features, seq_len)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = CNNModel()
    x = torch.randn(32, 24, 15)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
