"""
Focal Loss - Addresses class imbalance by down-weighting easy examples
Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor of size num_classes)
            gamma: Focusing parameter (default 2.0)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combines Focal Loss with optional physics-informed regularization"""
    
    def __init__(self, alpha=None, gamma=2.0, physics_weight=0.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.physics_weight = physics_weight
        
    def forward(self, outputs, targets, physics_pred=None):
        """
        Args:
            outputs: Model predictions (batch, num_classes)
            targets: Ground truth labels (batch,)
            physics_pred: Optional physics-based predictions for consistency
        """
        loss = self.focal_loss(outputs, targets)
        
        # Physics consistency (if provided)
        if physics_pred is not None and self.physics_weight > 0:
            physics_loss = F.mse_loss(
                torch.softmax(outputs, dim=1),
                torch.softmax(physics_pred, dim=1)
            )
            loss = loss + self.physics_weight * physics_loss
        
        return loss

if __name__ == "__main__":
    # Test
    criterion = FocalLoss(alpha=torch.tensor([0.25, 0.50, 0.75]), gamma=2.0)
    
    outputs = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    
    loss = criterion(outputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Compare with CE
    ce_loss = F.cross_entropy(outputs, targets)
    print(f"CE Loss: {ce_loss.item():.4f}")
