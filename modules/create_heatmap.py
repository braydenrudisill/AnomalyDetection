import torch

from modules.data import MVTEC_SYNTHETIC
from modules.models import TeacherNetwork


TEACHER_FEATURE_STATS_PATH = '/baldig/chemistry/2023_rp/Chemformer/pivot/models/teacher_stats.txt'


def main():
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    teacher = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    student = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)

    with open(TEACHER_FEATURE_STATS_PATH, 'r') as f:
        # mean1 std_dev1
        # mean2 std_dev2
        # ...
        means, std_devs = torch.tensor(list(zip(*[[float(num) for num in row.split(' ')] for row in f])), dtype=torch.float).to(device)

    teacher.load_state_dict(torch.load('models/teacher.pt'))
    student.load_state_dict(torch.load('models/students/2024-05-18T11:51:24.702777/student.pt'))

    s = 0.0018
    with open(MVTEC_SYNTHETIC / 'test/hole/0.txt', 'r') as f:
        sample_point_cloud = torch.tensor([[float(c) for c in line.strip().split(' ')] for line in f], dtype=torch.float, device=device) / s

    with torch.no_grad():
        teacher_features = teacher(sample_point_cloud.to(device))
        student_features = student(sample_point_cloud.to(device))

    diffs = (student_features - (teacher_features - means) / std_devs).pow(2).sum(dim=1).sqrt()

    print(diffs.shape)

    labeled_points = torch.cat([sample_point_cloud, diffs.unsqueeze(1)], dim=1)

    print(labeled_points.shape)

    print('Writing to file.')
    with open('anomalies_0.txt', 'w+') as f:
        f.writelines(' '.join(str(c.item()) for c in pt) + '\n' for pt in labeled_points)

    print('Done writing.')


if __name__ == '__main__':
    main()
