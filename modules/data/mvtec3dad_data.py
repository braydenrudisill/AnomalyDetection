import torch

from pathlib import Path
import cv2


MVTEC_SYNTHETIC = Path('pivotdata/synthetic_bagel')
MVTEC_PATH = Path('pivotdata/bagel')
MVTEC_DATASETS = ['bagel', ...]


def main():
    print(load_tiff(MVTEC_PATH / 'train' / 'good' / 'xyz' / '242.tiff').shape)


def load_tiff(path):
    """Returns a torch tensor containing all the point data in the TIFF point cloud file."""

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    print(image.shape)
    points = torch.flatten(torch.tensor(image), 0, 1)

    return points


if __name__ == '__main__':
    main()
