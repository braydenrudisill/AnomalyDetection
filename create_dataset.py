import numpy as np
import torch

from pathlib import Path

MODEL10_PATH = Path("../pivotdata/ModelNet10")

def main():
    points, faces = load_object(MODEL10_PATH / 'toilet/train/toilet_0100.off')
    print(points.shape, faces.shape)


def load_object(path: Path):
    """Returns a tuple of two numpy arrays containing all the point and face data in an OFF object file."""
    num_vertices, num_faces, _ = get_num_vertices_faces_cells(path)

    points = np.loadtxt(
        load_and_skip_prefixes(path, prefixes={'#', 'OFF'}),
        dtype = float,
        skiprows = 1,
        max_rows = num_vertices
    )
    faces = np.loadtxt(
        load_and_skip_prefixes(path, prefixes={'#', 'OFF'}),
        dtype = float,
        skiprows = num_vertices + 1,
        usecols = (1,2,3),
        max_rows = num_faces
    )

    return points, faces


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