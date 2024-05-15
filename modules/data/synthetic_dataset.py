import torch.utils.data
import numpy as np

import os
from pathlib import Path

from .modelnet10_data import load_object


class ModelNetDataset(torch.utils.data.Dataset):
    """ModelNet10 dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the OFF files.
        """
        self.root_dir = Path(root_dir)
        self.file_paths = list(self.root_dir.glob('*.off'))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene_path = self.root_dir / f'{idx:03}.txt'

        scene = load_object(scene_path)
        sample = {'scene': scene}

        return sample
