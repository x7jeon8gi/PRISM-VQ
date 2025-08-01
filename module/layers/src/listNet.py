import torch
import torch.nn.functional as F
import torch.nn as nn

class ListNetLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred, target):
        """
        pred: (batch_size,) 예측 수익률
        target: (batch_size,) 실제 수익률
        """
        # Temperature scaling으로 분포 sharpness 조절
        pred_prob = F.softmax(pred / self.temperature, dim=0)
        target_prob = F.softmax(target / self.temperature, dim=0)
        
        # KL divergence (simplified)
        loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-10))
        
        return loss