import numpy as np
from typing import List, Optional, Tuple, Union

from modules.data.mvtec3dad_data import MVTEC_PATH, SYNTHETIC_DATA_PATH, load_tiff

from random import randint
import torch
from tqdm import tqdm


# Original paper uses 64_000
NUM_POINTS_PER_CLOUD = 16_000


def main():
    files = MVTEC_PATH / 'test' / 'hole' / 'xyz'
    print(files)
    for i, file_path in tqdm(enumerate(files.glob('*.tiff')), desc="Generating point clouds"):
        cloud = load_tiff(file_path).to('cuda:0')
        # downsampled = farthest_point_sampling_naive(cloud, NUM_POINTS_PER_CLOUD)
        downsampled_idx = sample_farthest_points_naive(torch.tensor(cloud).unsqueeze(0), num_points=NUM_POINTS_PER_CLOUD)
        downsampled = cloud[downsampled_idx]
        new_path = SYNTHETIC_DATA_PATH / f'test/hole/{i}.txt'
        save_to_file(downsampled[0], new_path)


# TODO: Move to own file
def save_to_file(point_cloud, path):
    """Saves a point cloud to a text file where each line is the space separated coordinates of a point."""
    with open(path, 'w+') as f:
        f.writelines([' '.join(str(c.item()) for c in point) + '\n' for point in point_cloud])


def sample_farthest_points_naive(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor],
    num_points: Union[int, List, torch.Tensor],
    random_start_point: bool = False,
) -> torch.Tensor:
    """Returns the indices of the selected_points."""
    num_batches, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((num_batches,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (num_batches,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("Invalid lengths.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(num_points, int):
        num_points = torch.full((num_batches,), num_points, dtype=torch.int64, device=device)
    elif isinstance(num_points, list):
        num_points = torch.tensor(num_points, dtype=torch.int64, device=device)

    if num_points.shape[0] != num_batches:
        raise ValueError("K and points must have the same batch dimension")

    # Find max value of K
    max_K = torch.max(num_points)

    # List of selected indices from each batch element
    all_sampled_indices = []

    for n in range(num_batches):
        # Initialize an array for the sampled indices, shape: (max_K,)
        sample_idx_batch = torch.full(
            (max_K.item(),),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        # Initialize closest distances to inf, shape: (P,)
        # This will be updated at each iteration to track the closest distance of the
        # remaining points to any of the selected points
        closest_dists = points.new_full(
            (lengths[n],),
            float("inf"),
            dtype=torch.float32,
        )

        # Select a random point index and save it as the starting point
        selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
        sample_idx_batch[0] = selected_idx

        # If the pointcloud has fewer than K points then only iterate over the min
        k_n = min(lengths[n], num_points[n])

        # Iteratively select points for a maximum of k_n
        for i in range(1, k_n):
            # Find the distance between the last selected point
            # and all the other points. If a point has already been selected
            # its distance will be 0.0, so it will not be selected again as the max.
            dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

            # If closer than currently saved distance to one of the selected
            # points, then updated closest_dists
            closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

            # The aim is to pick the point that has the largest
            # nearest neighbour distance to any of the already selected points
            selected_idx = torch.argmax(closest_dists)
            sample_idx_batch[i] = selected_idx

        # Add the list of points for this batch to the final list
        all_sampled_indices.append(sample_idx_batch)

    all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

    return all_sampled_indices


if __name__ == '__main__':
    main()
