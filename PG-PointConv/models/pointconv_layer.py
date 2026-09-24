import torch
import torch.nn as nn
import torch.nn.functional as F


class PointConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Static weight MLP mapping spatial displacement (3) to weight matrix (in -> out)
        self.mlp_weight = nn.Sequential(
            nn.Linear(3, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels * out_channels)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = 1e-8

    def forward(self, new_xyz, xyz, features, neighbors):
        """
        new_xyz: [B, N, 3] (采样后的中心点坐标)
        xyz: [B, M, 3] (原始的稠密点云坐标)
        features: [B, M, C_in] (原始的稠密特征)
        neighbors: [B, N, K] (N个中心点在M个点中对应的邻域索引)
        """
        B, N, K = neighbors.shape
        batch_indices = torch.arange(B, device=xyz.device).view(-1, 1, 1).expand(-1, N, K)

        # 收集原点云中的邻居坐标和特征
        neighbor_xyz = xyz[batch_indices, neighbors]  # [B, N, K, 3]
        neighbor_feat = features[batch_indices, neighbors]  # [B, N, K, C_in]

        # 使用下采样后的 new_xyz 作为聚类中心计算相对位移，维度完美对齐
        center_xyz = new_xyz.unsqueeze(2)  # [B, N, 1, 3]
        rel_pos = neighbor_xyz - center_xyz  # [B, N, K, 3]

        # 计算核密度逆衰减因子 S(p_j) (使用高斯衰减)
        dist = torch.norm(rel_pos, dim=-1)
        density_weights = torch.exp(-dist)  # [B, N, K]
        density_norm = density_weights / (density_weights.sum(dim=-1, keepdim=True) + self.eps)

        # 生成静态空间权重
        static_w = self.mlp_weight(rel_pos)  # [B, N, K, C_in * C_out]
        static_w = static_w.view(B, N, K, self.in_channels, self.out_channels)

        return static_w, neighbor_feat, density_norm, rel_pos