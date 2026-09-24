import torch


def knn(x, k):
    """
    K-Nearest Neighbors using PyTorch cdist.
    x: [B, N, 3]
    return: [B, N, K] indices
    """
    dist = torch.cdist(x, x)
    _, idx = torch.topk(dist, k, dim=-1, largest=False)
    return idx


def farthest_point_sampling(xyz, npoint):
    """
    Pure PyTorch Farthest Point Sampling.
    xyz: [B, N, 3]
    return: [B, npoint] indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def gather_points(points, idx):
    """
    points: [B, N, C]
    idx: [B, M]
    return: [B, M, C]
    """
    B, M = idx.shape
    C = points.shape[2]
    batch_indices = torch.arange(B, device=points.device).unsqueeze(1).expand(-1, M)
    return points[batch_indices, idx, :]