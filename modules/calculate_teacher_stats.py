import torch

from tqdm import tqdm
from collections import deque

from modules.data import M10_SYNTHETIC_16K
from modules.models import TeacherNetwork, DecoderNetwork, KNNGraph


class RunningStats:
    """Computes a rolling mean and stddev with a window length of n to limit memory use."""
    def __init__(self, n):
        self.n = n
        self.data = deque([0.0] * n, maxlen=n)
        self.mean = self.variance = self.sdev = 0.0

    def add(self, x):
        n = self.n
        oldmean = self.mean
        goingaway = self.data[0]
        self.mean = newmean = oldmean + (x - goingaway) / n
        self.data.append(x)
        self.variance += (x - goingaway) * (
            (x - newmean) + (goingaway - oldmean)) / (n - 1)
        self.sdev = torch.sqrt(self.variance)


def main():
    """Calculates the mean and std_dev for each dimension of teacher features across the training dataset."""
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    teacher = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    decoder = DecoderNetwork(d_model, 128).to(device)

    teacher.eval()

    knn_graph = KNNGraph()

    teacher.load_state_dict(torch.load('models/teachers/2024-05-19T07:40:20.796113/teacher_125.pt'))

    r = RunningStats(10)
    for i in tqdm(range(500)):
        with open(M10_SYNTHETIC_16K / f'train/{i:03}.txt', 'r') as f:
            sample_point_cloud = torch.tensor([[float(c) for c in line.strip().split(' ')] for line in f], dtype=torch.float) / 0.015

        with torch.no_grad():
            output = teacher(sample_point_cloud.to(device))

        output_avg = torch.mean(output, 0)
        r.add(output_avg)
        if i % 100 == 0.0:
            print(i, "mean", r.mean)
            print(i, "sdev", r.sdev)

    print('Writing to file.')
    with open('../models/teachers/2024-05-19T07:40:20.796113/teacher_stats.txt', 'w+') as f:
        f.writelines(f'{mean} {std_dev}\n' for mean, std_dev in zip(r.mean, r.sdev))

    print('Done writing.')


if __name__ == '__main__':
    main()
