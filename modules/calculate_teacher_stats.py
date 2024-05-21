import torch

from tqdm import tqdm

from modules.data import M10_SYNTHETIC_16K, MVTEC_SYNTHETIC, PointCloudDataset, MVTEC_SCALING
from modules.models import TeacherNetwork, DecoderNetwork, KNNGraph


class RunningStats(torch.nn.Module):
    """Calculates the running mean and standard deviation for given samples."""
    def __init__(self, d_model, device):
        super().__init__()
        self.n = 0
        self.sum = torch.zeros(d_model).to(device)
        self.sum_squared = torch.zeros(d_model).to(device)

    def add(self, x):
        self.n += len(x)
        self.sum += torch.sum(x, dim=0)
        self.sum_squared += torch.sum(x.pow(2), dim=0)

    @property
    def standard_deviation(self):
        return torch.sqrt(((self.n * self.sum_squared) - self.sum.pow(2)) / (self.n * (self.n - 1)))

    @property
    def mean(self):
        return self.sum / self.n


def main():
    """Calculates the mean and std_dev for each dimension of teacher features across the training dataset."""
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    teacher = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    teacher.eval()

    teacher.load_state_dict(torch.load('models/teachers/2024-05-19T07:40:20.796113/teacher_225.pt'))

    stats = RunningStats(d_model, device)
    dataset = PointCloudDataset(root_dir=MVTEC_SYNTHETIC / 'train', scaling_factor=1/MVTEC_SCALING)

    for i, sample_point_cloud in enumerate(iter(dataset)):
        with torch.no_grad():
            output = teacher(sample_point_cloud.to(device))

        stats.add(output)

        if i % 100 == 0.0:
            print(i, "mean", stats.mean)
            print(i, "sdev", stats.standard_deviation)

    print('Writing to file.')
    with open('models/teachers/2024-05-19T07:40:20.796113/teacher_stats_225.txt', 'w+') as f:
        f.writelines(f'{mean} {std_dev}\n' for mean, std_dev in zip(stats.mean, stats.standard_deviation))

    print('Done writing.')


if __name__ == '__main__':
    main()
