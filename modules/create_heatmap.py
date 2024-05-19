import torch

from modules.data import MVTEC_SYNTHETIC
from modules.models import TeacherNetwork


def main():
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    teacher_stats_path = 'models/teachers/2024-05-19T07:40:20.796113/teacher_stats.txt'

    teacher = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    student = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)

    with open(teacher_stats_path, 'r') as f:
        # mean1 std_dev1
        # mean2 std_dev2
        # ...
        means, std_devs = torch.tensor(list(zip(*[[float(num) for num in row.split(' ')] for row in f])), dtype=torch.float).to(device)

    teacher.load_state_dict(torch.load('models/teachers/2024-05-19T07:40:20.796113/teacher_125.pt'))
    student.load_state_dict(torch.load('models/student/2024-05-19T10:24:12.761858/student.pt'))

    s = 0.0018
    with open(MVTEC_SYNTHETIC / 'test/3.txt', 'r') as f:
        sample_point_cloud = torch.tensor([[float(c) for c in line.strip().split(' ')] for line in f], dtype=torch.float, device=device) / s

    with torch.no_grad():
        teacher_features = teacher(sample_point_cloud.to(device))
        student_features = student(sample_point_cloud.to(device))

    diffs = (student_features - (teacher_features - means) / std_devs).pow(2).sum(dim=1).sqrt()

    print(diffs.shape)

    labeled_points = torch.cat([sample_point_cloud, diffs.unsqueeze(1)], dim=1)

    print(labeled_points.shape)

    print('Writing to file.')
    with open('anomalies_train.txt', 'w+') as f:
        f.writelines(' '.join(str(c.item()) for c in pt) + '\n' for pt in labeled_points)

    print('Done writing.')


if __name__ == '__main__':
    main()
