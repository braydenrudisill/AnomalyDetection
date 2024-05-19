import numpy as np

from pathlib import Path


MODEL10_PATH = Path('../pivotdata/ModelNet10')
M10_SYNTHETIC = MODEL10_PATH / '../64k_synthetic'
M10_SYNTHETIC_16k = MODEL10_PATH / '../synthetic_data'
M10_DATASETS = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def load_object(path: Path):
    """Returns a numpy array containing all the point data in an OFF object file."""
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

    try:
        return tuple(map(int, data_line.strip().split(' ')))
    except Exception as e:
        print(path, e)
        quit()


def load_and_skip_prefixes(path: Path, prefixes: set):
    """Loads a file as a generator skipping lines that start with given prefixes."""
    with open(path) as f:
        filtered = (line for line in f if not any(line.strip().startswith(prefix) for prefix in prefixes))
        for row in filtered:
            yield row
