import torch.utils.data

from pathlib import Path


class MVTecDataset(torch.utils.data.Dataset):
    """Downsampled MVTec dataset."""

    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the txt files.
        """
        self.root_dir = Path(root_dir)
        self.file_paths = list(self.root_dir.glob('*.txt'))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scene_path = self.file_paths[idx]

        with open(scene_path, 'r') as f:
            scene = torch.tensor([list(map(float, line.split(' '))) for line in f])

        # sample = {'scene': scene}
        return scene
