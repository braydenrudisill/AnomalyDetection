import unittest
from modules.models import TeacherNetwork, KNNGraph
from modules.pretrain_teacher import get_rf
import torch


device = torch.device('cuda:0')


class NetworkShapeTests(unittest.TestCase):
    def test_8nn_of_9x9_is_square(self):
        knn_graph = KNNGraph()
        xs = torch.linspace(-5, 5, steps=9)
        ys = torch.linspace(-5, 5, steps=9)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        points = torch.tensor(list(zip(x.flatten(), y.flatten())), dtype=torch.float)

        assert points[knn_graph(points, 8)][40].tolist() == torch.tensor(
                [[ 0.0000,  1.2500],
                [ 1.2500,  0.0000],
                [-1.2500,  0.0000],
                [ 0.0000, -1.2500],
                [-1.2500,  1.2500],
                [-1.2500, -1.2500],
                [ 1.2500, -1.2500],
                [ 1.2500,  1.2500]]
            ).tolist()

    def test_8rf2_of_9x9_is_square(self):
        knn_graph = KNNGraph()
        xs = torch.linspace(-5, 5, steps=9)
        ys = torch.linspace(-5, 5, steps=9)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        points = torch.tensor(list(zip(x.flatten(), y.flatten())), dtype=torch.float)

        knn = knn_graph(points, 8)

        rf = get_rf(knn, torch.tensor([40]), 2)

        assert points[rf].tolist() == torch.tensor(
            [[-2.5000, -2.5000],
            [-1.2500, -2.5000],
            [ 0.0000, -2.5000],
            [ 1.2500, -2.5000],
            [ 2.5000, -2.5000],
            [-2.5000, -1.2500],
            [-1.2500, -1.2500],
            [ 0.0000, -1.2500],
            [ 1.2500, -1.2500],
            [ 2.5000, -1.2500],
            [-2.5000,  0.0000],
            [-1.2500,  0.0000],
            [ 0.0000,  0.0000],
            [ 1.2500,  0.0000],
            [ 2.5000,  0.0000],
            [-2.5000,  1.2500],
            [-1.2500,  1.2500],
            [ 0.0000,  1.2500],
            [ 1.2500,  1.2500],
            [ 2.5000,  1.2500],
            [-2.5000,  2.5000],
            [-1.2500,  2.5000],
            [ 0.0000,  2.5000],
            [ 1.2500,  2.5000],
            [ 2.5000,  2.5000]],
        ).tolist()


if __name__ == '__main__':
    unittest.main()
