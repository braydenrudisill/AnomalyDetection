import numpy as np

from modules.data import MVTEC_PATH, MVTEC_SYNTHETIC, load_tiff, MODEL10_PATH, M10_DATASETS, M10_SYNTHETIC_32K, load_object

import random
import torch
from tqdm import tqdm


OBJ_PER_SCENE = 10

NUM_TRAIN_SCENES = 500
NUM_TEST_SCENES = 25

# Original paper uses 64_000
NUM_POINTS_PER_CLOUD = 16_000


def create_mvtec_dataset():
    gt = MVTEC_PATH / 'test' / 'crack' / 'gt'
    xyz = MVTEC_PATH / 'test' / 'crack' / 'xyz'
    for i, (gt_path, xyz_path) in tqdm(enumerate(zip(sorted(gt.glob('*.png')), sorted(xyz.glob('*.tiff')))), desc="Generating point clouds"):
        is_defect_cloud = load_tiff(gt_path).to('cuda:0')
        points = load_tiff(xyz_path).to('cuda:0')

        downsampled_idx = sample_farthest_points(points.unsqueeze(0), num_samples=NUM_POINTS_PER_CLOUD)[0]
        print(xyz_path, gt_path)
        print(downsampled_idx.shape, points.shape, is_defect_cloud.shape)
        new_points = points[downsampled_idx]
        new_labels = is_defect_cloud.unsqueeze(-1)[downsampled_idx]
        new_path = MVTEC_SYNTHETIC / f'test/{i}_gt.txt'
        print(new_path)
        save_to_file(torch.cat([new_points, new_labels], -1), new_path)


def create_model_10_dataset():
    for i in tqdm(range(NUM_TRAIN_SCENES), desc='Generating Training Scenes'):
        train_scene = generate_scene(NUM_POINTS_PER_CLOUD, 'train')
        save_to_file(train_scene[0], M10_SYNTHETIC_32K / f'train/{i:03}.txt', )

    for i in tqdm(range(NUM_TEST_SCENES), desc='Generating Testing Scenes'):
        test_scene = generate_scene(NUM_POINTS_PER_CLOUD, 'test')
        save_to_file(test_scene[0], M10_SYNTHETIC_32K / f'test/{i:02}.txt')


def generate_scene(n_points, dataset):
    """Generates a point cloud containing `OBJ_PER_SCENE1` objects.

    Each object is:
    1. Random selected from all .off files in the dataset
    2. Scaled to fit in (-1,1)
    3. Rotated by a random rotation matrix.
    4. Randomly transformed by (-3,3)

    The whole scene point cloud lies within (-3.5, 3.5).
    """
    objects = []
    for _ in tqdm(range(OBJ_PER_SCENE), desc="Filling scene with objects", leave=False):
        path = random.choice(list((MODEL10_PATH / random.choice(M10_DATASETS) / dataset).glob('*.off')))
        point_cloud = load_object(path)
        scaled_object = point_cloud / max(get_bounding_box_lengths(point_cloud))
        rotated_object = np.dot(scaled_object, get_random_rotation_matrix().T)
        transformed_object = rotated_object + (np.random.rand(3) * 6 - 3)
        objects.append(transformed_object)

    all_points = torch.tensor([pt for cloud in objects for pt in cloud]).unsqueeze(0)
    return sample_farthest_points(all_points, n_points)


def save_to_file(point_cloud, path):
    """Saves a point cloud to a text file where each line is the space separated coordinates of a point."""
    with open(path, 'w+') as f:
        f.writelines([' '.join(str(c.item()) for c in point) + '\n' for point in point_cloud])


def get_bounding_box_lengths(points):
    """Returns the dimensions of a point cloud."""
    x, y, z = zip(*points)
    return max(x) - min(x), max(y) - min(y), max(z) - min(z)


def get_random_rotation_matrix():
    """Generates a random rotation matrix"""
    random_matrix = np.random.randn(3, 3)
    q, r = np.linalg.qr(random_matrix)
    q *= np.random.choice([-1,1], 3)

    return q


def sample_farthest_points(points: torch.Tensor, num_samples: int, random_start_point: bool = False) -> torch.Tensor:
    """Returns the indices of the selected_points."""
    num_batches, num_points, dimensions = points.shape
    device = points.device

    all_sampled_indices = []

    for batch in range(num_batches):
        sample_idx_batch = torch.full(
            (num_samples,),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        # Initialize the closest distances to inf, shape: (num_points,)
        closest_dists = points.new_full(
            (num_points,),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        selected_idx = random.randint(0, num_points - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the cloud has fewer points than sample size then get all points.
        num_to_sample = min(num_points, num_samples)

        for i in tqdm(range(1, num_to_sample), desc='Sampling'):
            dist = points[batch, selected_idx, :] - points[batch, :num_points, :]

            closest_dists = torch.min(dist.pow(2).sum(-1), closest_dists)
            selected_idx = torch.argmax(closest_dists)

            sample_idx_batch[i] = selected_idx

        all_sampled_indices.append(sample_idx_batch)

    return torch.stack(all_sampled_indices, dim=0)


if __name__ == '__main__':
    # create_model_10_dataset()
    create_mvtec_dataset()
