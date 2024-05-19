import torch
from torch.utils.data import DataLoader

from modules.models import TeacherNetwork
from modules.data import PointCloudDataset, MVTEC_SYNTHETIC
from modules.trainer import Trainer
from modules.loss import EuclidianFeatureDistance


def main():
    d = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:0')
    teacher_path = 'models/teachers/2024-05-19T07:40:20.796113/teacher_125.pt'
    teacher_stats_path = 'models/teachers/2024-05-19T07:40:20.796113/teacher_stats.txt'

    train_dataset = PointCloudDataset(root_dir=MVTEC_SYNTHETIC/'train', scaling_factor=1/0.0018)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    test_dataset = PointCloudDataset(root_dir=MVTEC_SYNTHETIC/'test', scaling_factor=1/0.0018)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    trainer = StudentTrainer(d, k, num_blocks, device, teacher_path, teacher_stats_path, train_dataloader, test_dataloader)
    trainer.run()


class StudentTrainer(Trainer):
    def __init__(self, d_model, k, num_res_blocks, device, teacher_path, teacher_stats_path, train_dataloader, test_dataloader):
        super().__init__('student', 100, device, train_dataloader, test_dataloader)

        self.teacher = TeacherNetwork(d_model, k, num_res_blocks, device=device).to(device)
        self.student = TeacherNetwork(d_model, k, num_res_blocks, device=device).to(device)
        self.criterion = EuclidianFeatureDistance().to(device)

        self.teacher.load_state_dict(torch.load(teacher_path))

        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=1e-3,
            weight_decay=1e-5)

        # Stats file format:
        # mean1 std_dev1
        # mean2 std_dev2
        # ...
        with open(teacher_stats_path, 'r') as f:
            self.means, self.std_devs = torch.tensor(list(zip(*[[float(num) for num in row.split(' ')] for row in f])),
                                           dtype=torch.float).to(self.device)

    def predict_and_score(self, batch: torch.Tensor) -> torch.Tensor:
        point_cloud = batch[0].to(self.device)

        with torch.no_grad():
            descriptors = self.teacher(point_cloud)
        predicted_descriptors = self.student(point_cloud)

        loss = self.criterion(predicted_descriptors, descriptors, self.means, self.std_devs)
        return loss

    def save_models(self, tag) -> None:
        torch.save(self.student.state_dict(), f'{self.model_path}/student.pt')


if __name__ == '__main__':
    main()
