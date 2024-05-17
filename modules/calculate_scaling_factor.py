from torch.utils.data import DataLoader

from modules.data.modelnet10_data import SYNTHETIC_DATA_PATH
from modules.data.synthetic_dataset import ModelNetDataset
from models.knn_graph import KNNGraph
from tqdm import tqdm

import torch

import itertools


def main():
    k = 8
    train_ds = ModelNetDataset(SYNTHETIC_DATA_PATH / 'train')
    test_ds = ModelNetDataset(SYNTHETIC_DATA_PATH / 'test')
    knn_graph = KNNGraph()

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_dataloder = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)

    num_points = 0
    total_distance = 0
    for cloud in tqdm(itertools.chain(train_dataloader, test_dataloder), position=0, leave=True):
        cloud = cloud[0]
        knn = knn_graph(cloud, k)
        for p_i, p in enumerate(cloud):
            total_distance += sum(torch.linalg.norm(p - cloud[neighbor]) for neighbor in knn[p_i])
        num_points += len(cloud)

        print(total_distance / (num_points*k))


if __name__ == '__main__':
    main()
