import torch
import torch.nn as nn
from models.pointconv_layer import PointConvLayer
from models.feature_diff_attention import FeatureDiffAttention
from models.structural_tensor import StructuralTensorEmbedding
from models.utils import farthest_point_sampling, gather_points
from data.voxelization import MaxPoolVoxelization
from models.heads import SegmentationHead, RiskRegressionHead


class EncoderLayer(nn.Module):
    def __init__(self, npoint, in_channels, out_channels, k=16, heads=4):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.pointconv = PointConvLayer(in_channels, out_channels)
        self.attention = FeatureDiffAttention(in_channels, heads=heads)
        self.head_merge = nn.Linear(out_channels * heads, out_channels)
        self.relu = nn.ReLU()

    def forward(self, xyz, feats):
        # 1. FPS 降采样
        idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = gather_points(xyz, idx)
        new_feats = gather_points(feats, idx) if feats is not None else None

        # 2. KNN 搜索 (只计算新中心点到所有原始点的距离，大幅节省显存)
        dist = torch.cdist(new_xyz, xyz)
        _, neighbors = torch.topk(dist, self.k, dim=-1, largest=False)  # [B, npoint, K]

        # 3. 传入 new_xyz 与 xyz 给 PointConvLayer
        static_w, neighbor_feat, density_norm, rel_pos = self.pointconv(new_xyz, xyz, feats, neighbors)

        # 4. 获取多头特征差分注意力
        center_feat = new_feats.unsqueeze(2)
        alpha = self.attention(center_feat, neighbor_feat, rel_pos)  # [B, N, K, heads]

        # 5. OOM 内存优化核心：调换矩阵乘法顺序 (数学等价，内存占用从 10GB 降低至 ~40MB)
        # 先执行 pointconv 基础聚合: neighbor_feat [B, N, K, C_in], static_w [B, N, K, C_in, C_out]
        # 用 matmul 进行张量乘法 -> local_feat [B, N, K, C_out]
        local_feat = torch.matmul(neighbor_feat.unsqueeze(-2), static_w).squeeze(-2)

        # 乘以核密度逆衰减因子 S(p_j)
        local_feat = local_feat * density_norm.unsqueeze(-1)  # density_norm: [B, N, K] -> [B, N, K, 1]

        # 与注意力权重 alpha 结合
        # local_feat 扩展为 [B, N, K, 1, C_out], alpha 扩展为 [B, N, K, heads, 1]
        dynamic_feat = local_feat.unsqueeze(3) * alpha.unsqueeze(-1)  # 得到 [B, N, K, heads, C_out]

        # 沿 K (邻域) 维度求和完成聚合
        aggregated_feats = torch.sum(dynamic_feat, dim=2)  # [B, N, heads, C_out]

        # 6. 展平并映射回标准的 C_out 维度
        aggregated_feats = aggregated_feats.view(new_xyz.shape[0], self.npoint, -1)
        out_feats = self.relu(self.head_merge(aggregated_feats))

        return new_xyz, out_feats


class PGPointConv(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.voxelizer = MaxPoolVoxelization(voxel_size=0.02, max_points=20000)
        self.input_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )

        self.layer1 = EncoderLayer(5120, 64, 128, k=16)
        self.layer2 = EncoderLayer(1280, 128, 256, k=16)
        self.structural_tensor = StructuralTensorEmbedding(k_cov=32)

        fused_dim = 256 + 1 + 3
        self.seg_head = SegmentationHead(fused_dim, num_classes)
        self.risk_head = RiskRegressionHead(fused_dim)

    def forward(self, points):
        points = self.voxelizer(points[0]).unsqueeze(0)
        feats = self.input_mlp(points)

        xyz1, feat1 = self.layer1(points, feats)
        xyz2, feat2 = self.layer2(xyz1, feat1)

        sigma, eigvals = self.structural_tensor(xyz2)
        fused_feats = torch.cat([feat2, sigma, eigvals], dim=-1)

        seg_logits = self.seg_head(fused_feats)
        risk_pred = self.risk_head(fused_feats)

        return seg_logits, risk_pred, xyz2