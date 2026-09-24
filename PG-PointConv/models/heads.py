import torch
import torch.nn as nn

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, feats):
        # feats: [B, N, C]
        return self.mlp(feats)

class RiskRegressionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, feats):
        # feats: [B, N, C]
        return self.mlp(feats)