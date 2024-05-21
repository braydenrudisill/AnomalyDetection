import unittest
import torch
from modules.calculate_teacher_stats import RunningStats

class StatsTests(unittest.TestCase):
    def test_standard_deviation_of_repeated_list_is_0(self):
        stats = RunningStats(2, torch.device('cpu'))
        stats.add(torch.tensor([[0,1], [0,1]]))

        assert all(stats.standard_deviation == 0)

    def test_non_zero_standard_deviation(self):
        stats = RunningStats(2, torch.device('cpu'))
        stats.add(torch.tensor([[0, 4], [2, 0], [0, 0], [0, 0]]))
        assert stats.standard_deviation.tolist() == [1, 2]


if __name__ == '__main__':
    unittest.main()
