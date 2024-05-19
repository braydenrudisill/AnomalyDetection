import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from modules.models import TeacherNetwork
from modules.data import ModelNetDataset
from modules.data.mvtec3dad_data import SYNTHETIC_DATA_PATH


TEACHER_FEATURE_STATS_PATH = '/baldig/chemistry/2023_rp/Chemformer/pivot/models/teacher_stats.txt'
TEACHER_MODEL_PATH = '/baldig/chemistry/2023_rp/Chemformer/pivot/models/teacher.pt'


def main():
    d = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:0')
    criterion = StudentLoss().to(device)
    teacher_model = TeacherNetwork(d, k, num_blocks, device).to(device)
    student_model = TeacherNetwork(d, k, num_blocks, device).to(device)
    teacher_model.load_state_dict(torch.load(TEACHER_MODEL_PATH))

    writer = SummaryWriter()
    date = datetime.now().isoformat()
    model_path = Path(f"/baldig/chemistry/2023_rp/Chemformer/pivot/models/students/{date}")
    model_path.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3, weight_decay=1e-6)

    with open(TEACHER_FEATURE_STATS_PATH, 'r') as f:
        # mean1 std_dev1
        # mean2 std_dev2
        # ...
        means, std_devs = torch.tensor(list(zip(*[[float(num) for num in row.split(' ')] for row in f])), dtype=torch.float).to(device)

    train_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH / 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    s = 0.0018
    scaling_factor = 1 / s
    for epoch in range(100):
        print(f'EPOCH: {epoch}')
        epoch_loss = 0
        for step, data in tqdm(enumerate(train_dataloader), desc='Training'):
            point_cloud = data[0].to(device) * scaling_factor

            with torch.no_grad():
                descriptors = teacher_model(point_cloud)

            predicted_descriptors = student_model(point_cloud)

            loss = criterion(predicted_descriptors, descriptors, means, std_devs)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch Loss: {epoch_loss / len(train_dataloader)}')
        if writer is not None:
            writer.add_scalar('Epoch Loss/Total', epoch_loss / len(train_dataloader), epoch)

        if (epoch + 1) % 25 == 0:
            torch.save(student_model.state_dict(), f'{model_path}/student.pt')


def sample_loss():
    criterion = StudentLoss()
    f1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    f2 = torch.tensor([[0, 1, 2, 10], [4, 5, 9, 22]])
    means = torch.zeros(4)
    std_devs = torch.ones(4)
    std_devs[3] = 2
    print(criterion(f1, f2, means, std_devs))


class StudentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features, means, std_devs):
        """Takes the average feature-wise L2 difference between the features from the two point clouds.

        Student_features: (n, d)
        teacher_features: (n, d)
        means: (d,)
        std_devs: (d,)
        """

        n_points = len(teacher_features)

        total_distance = torch.sum((student_features - (teacher_features - means) / std_devs).pow(2).sum(dim=1).sqrt())

        return total_distance / n_points


if __name__ == '__main__':
    main()