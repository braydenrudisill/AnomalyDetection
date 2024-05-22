from pathlib import PosixPath

import torch

from modules.data import MVTEC_SYNTHETIC, PointCloudDataset, MVTEC_SCALING
from modules.models import TeacherNetwork


def main():
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    student_path = 'models/student/2024-05-21T05:48:21.346390/student.pt'
    teacher_path = 'models/teacher/2024-05-20T22:23:41.177392/teacher_225.pt'
    teacher_stats_path = 'models/teacher/2024-05-20T22:23:41.177392/teacher_stats_225.txt'

    teacher = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    student = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)

    with open(teacher_stats_path, 'r') as f:
        means, std_devs = torch.tensor([list(map(float, line.split(' '))) for line in f]).T.to(device)

    teacher.load_state_dict(torch.load(teacher_path))
    student.load_state_dict(torch.load(student_path))

    dataset = PointCloudDataset(root_dir=MVTEC_SYNTHETIC / 'test', scaling_factor=MVTEC_SCALING)

    sample_point_cloud = dataset[22][0].to(device)[:, :3]

    with torch.no_grad():
        teacher_features = teacher(sample_point_cloud.to(device))
        student_features = student(sample_point_cloud.to(device))

    diffs = (student_features - (teacher_features - means) / std_devs).pow(2).sum(dim=1).sqrt()
    labeled_points = torch.cat([sample_point_cloud, diffs.unsqueeze(1)], dim=1)

    print('Writing to file.')
    with open('anomalies_train.txt', 'w+') as f:
        f.writelines(' '.join(str(c.item()) for c in pt) + '\n' for pt in labeled_points)

    print('Done writing.')


if __name__ == '__main__':
    main()
