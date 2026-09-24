import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDiffAttention(nn.Module):
    def __init__(self, channels, heads=4, temperature=0.1):
        super().__init__()
        self.heads = heads
        self.temperature = temperature
        self.eps = 1e-8

        self.mlp = nn.Sequential(
            nn.Linear(channels + 3, channels // 2),
            nn.BatchNorm1d(channels // 2),  # Applying 1d over combined N*K
            nn.ReLU(),
            nn.Linear(channels // 2, heads)
        )

    def forward(self, center_feat, neighbor_feat, rel_pos):
        """
        center_feat: [B, N, 1, C]
        neighbor_feat: [B, N, K, C]
        rel_pos: [B, N, K, 3]
        return: alpha [B, N, K, heads]
        """
        B, N, K, C = neighbor_feat.shape

        # Calculate feature difference |f_j - f_i|
        delta_f = torch.abs(neighbor_feat - center_feat)  # [B, N, K, C]

        # Concatenate diff and relative position
        x = torch.cat([delta_f, rel_pos], dim=-1)  # [B, N, K, C+3]
        x = x.view(-1, C + 3)

        # MLP
        alpha_logits = self.mlp(x).view(B, N, K, self.heads)

        # Softmax with temperature tau=0.1
        alpha = F.softmax(alpha_logits / self.temperature, dim=2)  # [B, N, K, heads]
        return alpha