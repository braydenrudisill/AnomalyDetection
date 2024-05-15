import numpy as np
import torch

import random
from pathlib import Path

MODEL10_PATH = Path("../pivotdata/ModelNet10").resolve()
DATASETS = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

OBJ_PER_SCENE = 10
TRAIN_POINTS = 1000


def main():
    train_s = generate_scene(TRAIN_POINTS, 'train')
    print(s1[:3])


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
    for _ in range(OBJ_PER_SCENE):
        path = random.choice(list((MODEL10_PATH / random.choice(DATASETS) / dataset).glob('*.off')))
        point_cloud = load_object(path)
        scaled_object = point_cloud / max(get_bounding_box_lengths(point_cloud))
        rotated_object = np.dot(scaled_object, get_random_rotation_matrix().T)
        transformed_object = rotated_object + (np.random.rand(3) * 6 - 3)
        objects.append(transformed_object)

    all_points = [pt for cloud in objects for pt in cloud]
    return fps(all_points, n_points)


def fps(points, n_samples):
    """Samples `n_samples` points from a given point cloud using Farthest Point Sampling."""
    points = np.array(points)
    points_left = np.arange(len(points))
    sample_inds = np.zeros(n_samples, dtype='int')
    dists = np.ones_like(points_left) * float('inf')
    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected)

    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        dist_to_last_added_point = ((points[last_added] - points[points_left]) ** 2).sum(-1)
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

def fps2(points, n_samples):
    """Samples `n_samples` points from a given point cloud using Farthest Point Sampling."""
    points = np.array(points)
    n_points = len(points)
    sample_inds = np.zeros(n_samples, dtype=int)
    dists = np.full(n_points, np.inf)
    sample_inds[0] = np.random.randint(n_points)

    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        dist_to_last_added_point = np.sum((points[last_added] - points) ** 2, axis=1)
        dists = np.minimum(dists, dist_to_last_added_point)
        sample_inds[i] = np.argmax(dists)

    return points[sample_inds]


def get_bounding_box_lengths(points):
    """Returns the dimensions of a point cloud."""
    x, y, z = zip(*points)
    return max(x) - min(x), max(y) - min(y), max(z) - min(z)


def get_random_rotation_matrix():
    """Generates a random rotation matrix"""
    z = np.random.randn(3, 3)
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=0, axis2=1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[0, :] *= np.linalg.det(rot)[..., None]
    return rot


def load_object(path: Path):
    """Returns a tuple of two numpy arrays containing all the point data in an OFF object file."""
    num_vertices, _, _ = get_num_vertices_faces_cells(path)

    points = np.loadtxt(
        load_and_skip_prefixes(path, prefixes={'#', 'OFF'}),
        dtype=float,
        skiprows=1,
        max_rows=num_vertices
    )

    return points


def get_num_vertices_faces_cells(path: Path) -> tuple:
    """Returns a tuple describing the number of vertices, faces, and cells in an OFF object file."""
    data_line = next(load_and_skip_prefixes(path, prefixes={'#', 'OFF'}))
    return tuple(map(int, data_line.strip().split(' ')))


def load_and_skip_prefixes(path: Path, prefixes: set):
    """Loads a file as a generator skipping lines that start with given prefixes."""
    with open(path) as f:
        filtered = (line for line in f if not any(line.strip().startswith(prefix) for prefix in prefixes))
        for row in filtered:
            yield row


if __name__ == '__main__':
    main()