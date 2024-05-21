from torch.utils.data import DataLoader

from tqdm import tqdm

from modules.data import PointCloudDataset, MODEL10_PATH, M10_SYNTHETIC_16K, M10_SCALING_16

import itertools


def main():
    train_ds = PointCloudDataset(M10_SYNTHETIC_16K / 'train', M10_SCALING_16)
    test_ds = PointCloudDataset(M10_SYNTHETIC_16K / 'test', M10_SCALING_16)

    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)
    for i, cloud in tqdm(enumerate(itertools.chain(train_dataloader, test_dataloader)), position=0, leave=True):
        try:
            assert len(cloud[0]) == 16000, (len(cloud[0]), i)
        except AssertionError as e:
            print(e)


if __name__ == '__main__':
    main()
