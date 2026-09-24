import torch
import torch.nn as nn
from models.utils import knn


class StructuralTensorEmbedding(nn.Module):
    def __init__(self, k_cov=32, eps_cov=1e-6):
        super().__init__()
        self.k_cov = k_cov
        self.eps_cov = eps_cov

    def forward(self, xyz):
        """
        xyz: [B, N, 3]
        return: sigma [B, N, 1], eigvals [B, N, 3]
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # Find local neighborhood for covariance (k_cov=32)
        neighbors = knn(xyz, self.k_cov)
        batch_indices = torch.arange(B, device=device).view(-1, 1, 1).expand(-1, N, self.k_cov)
        neighbor_xyz = xyz[batch_indices, neighbors]  # [B, N, K, 3]

        # Calculate Covariance
        mean_xyz = neighbor_xyz.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
        centered = neighbor_xyz - mean_xyz  # [B, N, K, 3]

        # cov = (centered^T * centered) / K
        cov = torch.matmul(centered.transpose(-1, -2), centered) / self.k_cov  # [B, N, 3, 3]

        # PCA: Eigen decomposition
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigvals, _ = torch.linalg.eigh(cov)  # [B, N, 3]

        # Curvature sigma_i = lambda_3 / (lambda_1 + lambda_2 + lambda_3 + eps)
        # Since ascending, lambda_3 is index 0
        lambda_3 = eigvals[:, :, 0]
        sum_eig = eigvals.sum(dim=-1) + self.eps_cov
        sigma = (lambda_3 / sum_eig).unsqueeze(-1)  # [B, N, 1]

        # Reverse eigvals to descending order for concatenation [lambda_1, lambda_2, lambda_3]
        eigvals_desc = torch.flip(eigvals, dims=[-1])

        return sigma, eigvals_desc