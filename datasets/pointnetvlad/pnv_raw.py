import numpy as np
import os
import sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from config import PARAMS 
from datasets.base_datasets import PointCloudLoader
import torch


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray

        file_path = os.path.join(file_pathname)

        if PARAMS.protocol == 'usyd':
            pc = np.fromfile(file_path, dtype=np.float32).reshape([-1, 4])
            pc = pc[:, :3]
            pc = pc[np.linalg.norm(pc[:, :3], axis=1) < PARAMS.max_distance]
            # shuffle points in case they are randomly subsampled later
            np.random.shuffle(pc)

        else:
            pc = np.fromfile(file_path, dtype=np.float64)
            pc = np.float32(pc)
            # coords are within -1..1 range in each dimension
            pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        return pc
