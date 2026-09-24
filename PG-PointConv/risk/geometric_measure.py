import torch
import numpy as np


class GeometricMeasure:
    def __init__(self):
        pass

    def extract(self, regions, full_xyz):
        """
        regions: list of [M, 3] obstacle point arrays
        full_xyz: [N, 3] all points in scene to calculate clearance
        """
        delta_h_list = []
        w_list = []

        if not regions:
            return torch.tensor([]), torch.tensor([])

        full_xyz_np = full_xyz.cpu().numpy()

        for reg in regions:
            # delta h: Max Z - Min Z in the local cluster
            max_z = np.max(reg[:, 2])
            min_z = np.min(reg[:, 2])
            delta_h = max_z - min_z

            # w: effective traffic width (simplified as bounding box dimension for clearance)
            # In practical aging housing, clearance is distance from obstacle to nearest wall
            # Here we approximate bounding box footprint as occupied width
            max_x = np.max(reg[:, 0])
            min_x = np.min(reg[:, 0])
            max_y = np.max(reg[:, 1])
            min_y = np.min(reg[:, 1])
            occupied_width = max(max_x - min_x, max_y - min_y)

            # Assume a standard 1.2m hallway, effective width is hallway - occupied
            effective_w = max(1.2 - occupied_width, 0.0)

            delta_h_list.append(delta_h)
            w_list.append(effective_w)

        return torch.tensor(delta_h_list, dtype=torch.float32), torch.tensor(w_list, dtype=torch.float32)