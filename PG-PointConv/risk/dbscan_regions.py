import torch
from sklearn.cluster import DBSCAN
import numpy as np


class RiskRegionBuilder:
    def __init__(self, eps=0.05, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_algo = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def build(self, xyz, seg_preds):
        """
        xyz: [N, 3] points
        seg_preds: [N] predicted class indices
        Assuming class 3 and 4 are obstacles (scattered debris, low thresholds)
        """
        device = xyz.device
        xyz_np = xyz.cpu().numpy()
        seg_np = seg_preds.cpu().numpy()

        # Filter obstacle points
        obs_mask = (seg_np == 3) | (seg_np == 4)
        obs_points = xyz_np[obs_mask]

        if len(obs_points) == 0:
            return [], []

        labels = self.cluster_algo.fit_predict(obs_points)

        regions = []
        masks = []

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue  # Noise
            cluster_mask = (labels == cluster_id)
            regions.append(obs_points[cluster_mask])

            # Map back to original indices for gradient matching if needed
            full_mask = torch.zeros(len(xyz), dtype=torch.bool, device=device)
            obs_indices = torch.nonzero(torch.tensor(obs_mask, device=device)).squeeze()
            if obs_indices.dim() == 0:
                obs_indices = obs_indices.unsqueeze(0)
            cluster_indices = obs_indices[torch.tensor(cluster_mask, device=device)]
            full_mask[cluster_indices] = True
            masks.append(full_mask)

        return regions, masks