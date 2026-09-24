import torch


class MaxPoolVoxelization:
    def __init__(self, voxel_size=0.02, max_points=20000):
        self.voxel_size = voxel_size
        self.max_points = max_points

    def __call__(self, points):
        # points: [N, 3]
        device = points.device
        coords = torch.round(points / self.voxel_size).to(torch.int32)

        # 1. 按照 Z 轴（高度）降序排序，这样排在前面的点就是每个体素中最高的点
        z_values = points[:, 2]
        sorted_indices = torch.argsort(z_values, descending=True)
        points_sorted = points[sorted_indices]
        coords_sorted = coords[sorted_indices]

        # 2. 获取去重后的体素坐标和原坐标到唯一体素坐标的映射 (inverse_indices)
        # 此时 return_inverse=True 会让 torch.unique 返回 (唯一值, 逆向索引) 两个张量
        unique_coords, inverse_indices = torch.unique(coords_sorted, dim=0, return_inverse=True)

        # 3. 提取每个唯一体素中第一次出现的点（即 Z 最大的点）
        num_unique = unique_coords.shape[0]
        first_occurrence = torch.empty(num_unique, dtype=torch.long, device=device)

        # 因为 scatter_ 是覆盖写入，我们倒序遍历并写入索引，最后留下的必定是最小的索引（也就是第一次出现的最高点）
        N = coords_sorted.shape[0]
        reverse_indices = torch.arange(N - 1, -1, -1, device=device)
        first_occurrence.scatter_(0, inverse_indices[reverse_indices], reverse_indices)

        voxelized_points = points_sorted[first_occurrence]

        # 4. 填充或下采样到固定点数 (论文中为 20,000)
        num_pts = voxelized_points.shape[0]
        if num_pts >= self.max_points:
            # 如果点数过多，随机下采样至 max_points
            indices = torch.randperm(num_pts, device=device)[:self.max_points]
            voxelized_points = voxelized_points[indices]
        else:
            # 如果点数不足，用 0 填充边界 (为了批量处理)
            padding = torch.zeros((self.max_points - num_pts, 3), device=device)
            voxelized_points = torch.cat([voxelized_points, padding], dim=0)

        return voxelized_points