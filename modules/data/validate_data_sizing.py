from torch.utils.data import DataLoader

from modelnet10_data import SYNTHETIC_DATA_PATH
from synthetic_dataset import ModelNetDataset
from models.teacher_network import knn_graph
from tqdm import tqdm
import random

from modelnet10_data import MODEL10_PATH, DATASETS, load_object
import torch

import itertools

def get_num_points(path):
    return len(load_object(path))

def main():
    # print(min((MODEL10_PATH / random.choice(DATASETS) / 'train').glob('*.off'), key=get_num_points))
    train_ds = ModelNetDataset(SYNTHETIC_DATA_PATH / 'train')
    test_ds = ModelNetDataset(SYNTHETIC_DATA_PATH / 'test')

    s = 0.0633
    scaling_factor = 1 / s

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_dataloder = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)
    for i, cloud in tqdm(enumerate(itertools.chain(train_dataloader, test_dataloder)), position=0, leave=True):
        with open('sample_scaled_scene.txt', 'w') as f:
            f.writelines([' '.join([str(coord.item()) for coord in pt]) + '\n' for pt in cloud[0] * scaling_factor])
        quit()
        # try:
        #     assert len(cloud[0]) == 16000, (len(cloud[0]), i)
        # except AssertionError as e:
        #     print(e)


if __name__ == '__main__':
    main()
