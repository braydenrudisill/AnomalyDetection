import torch
from torch.utils.data import DataLoader

from modules.models import TeacherNetwork, KNNGraph, DecoderNetwork
from modules.data import PointCloudDataset, M10_SYNTHETIC_16k
from modules.trainer import Trainer
from modules.loss import ChamferDistance


NUM_FEATURE_SAMPLES = 16


def main():
    d_model = 64
    k = 8
    num_res_blocks = 4
    num_decoded_points = 128
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    train_dataset = PointCloudDataset(root_dir=M10_SYNTHETIC_16k / 'train', scaling_factor=1/0.015)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    test_dataset = PointCloudDataset(root_dir=M10_SYNTHETIC_16k / 'test', scaling_factor=1/0.015)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    trainer = TeacherPretrainer(d_model, k, num_res_blocks, device, num_decoded_points, train_dataloader, test_dataloader)
    trainer.run()


class TeacherPretrainer(Trainer):
    def __init__(self, d_model, k, num_res_blocks, device, num_decoded_points, train_dataloader, test_dataloader):
        super().__init__('teacher', device, train_dataloader, test_dataloader)

        self.k = k
        self.num_res_blocks = num_res_blocks

        self.teacher = TeacherNetwork(d_model, k, num_res_blocks, device=device).to(device)
        self.decoder = DecoderNetwork(d_model, num_decoded_points).to(device)
        self.criterion = ChamferDistance().to(device)
        self.knn_graph = KNNGraph().to(device)

        self.optimizer = torch.optim.Adam(
            list(self.teacher.parameters()) + list(self.decoder.parameters()),
            lr=1e-3,
            weight_decay=1e-6)

    def predict_and_score(self, batch) -> torch.Tensor:
        point_cloud = batch[0].to(self.device)
        knn = self.knn_graph(point_cloud, self.k)

        features = self.teacher(point_cloud, knn)
        sampled_indices = torch.randint(0, len(point_cloud), (NUM_FEATURE_SAMPLES,))

        all_losses = []
        reconstructed_points = self.decoder(features[sampled_indices])
        for i, pts in zip(sampled_indices, reconstructed_points):
            # 2 LFA per resnet block
            rf_indices = get_rf(knn, torch.tensor([i]).to(self.device), self.num_res_blocks * 2)
            rf = point_cloud[rf_indices]
            rf_centered = rf - torch.mean(rf, dim=0)
            loss = self.criterion(rf_centered, pts.reshape(-1, 3))
            all_losses.append(loss)

        return sum(all_losses) / NUM_FEATURE_SAMPLES

    def save_models(self, tag):
        torch.save(self.teacher.state_dict(), f'{self.model_path}/teacher_{tag}.pt')
        torch.save(self.decoder.state_dict(), f'{self.model_path}/decoder_{tag}.pt')


def get_rf(knn, i, n):
    result = i
    for _ in range(n):
        result = torch.unique(torch.cat([result, torch.flatten(knn[result])], dim=0))

    return result


if __name__ == '__main__':
    main()
