import unittest
from modules.models import TeacherNetwork
import torch

cuda0 = torch.device('cuda:0')
class NetworkShapeTests(unittest.TestCase):
    def test_teach_network_output_is_shaped_b_by_n_by_d(self):
        n = 16000
        d = 64
        model = TeacherNetwork(d, 5, n, 1, device=cuda0).to(cuda0)
        with open('/baldig/chemistry/2023_rp/Chemformer/pivotdata/synthetic_data/train/010.txt', 'r') as f:
            p = [list(map(float, line.split(' '))) for line in f]
            points = torch.tensor([p]).to(cuda0)
        output = model(points)
        self.assertEqual(output.shape, (1, n, d))


if __name__ == '__main__':
    unittest.main()
