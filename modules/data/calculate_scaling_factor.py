from torch.utils.data import DataLoader

from modules.data import MVTEC_SYNTHETIC, PointCloudDataset, M10_SYNTHETIC_16K, M10_SYNTHETIC
from modules.models import KNNGraph
from tqdm import tqdm

import torch


def main():
    k = 8
    train_ds = PointCloudDataset(M10_SYNTHETIC / 'train', 1)
    knn_graph = KNNGraph()

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    num_points = 0
    total_distance = 0
    for cloud in tqdm(train_dataloader, position=0, leave=True):
        cloud = cloud[0]
        knn = knn_graph(cloud, k)
        for p_i, p in enumerate(cloud):
            total_distance += sum(torch.linalg.norm(p - cloud[neighbor]) for neighbor in knn[p_i])
        num_points += len(cloud)

        print(total_distance / (num_points*k))


if __name__ == '__main__':
    main()
