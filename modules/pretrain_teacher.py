import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from models.teacher_network import TeacherNetwork, SharedMLP, LocalFeatureAggregation, ResBlock
from models.knn_graph import KNNGraph
from models.decoder_network import DecoderNetwork
from modules.data.synthetic_dataset import ModelNetDataset
from modules.data.modelnet10_data import SYNTHETIC_DATA_PATH_16K


NUM_FEATURE_SAMPLES = 16




class ChamferDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        diff = a[:, None, :] - b[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        min_dist, _ = torch.min(dist_matrix, dim=1)
        chamfer_dist = torch.sum(min_dist)

        return chamfer_dist


def get_rf(knn, i, n):
    result = knn[i]
    for _ in range(n):
        result = torch.unique(torch.flatten(knn[result]))
    return result


def pretrain_teacher_network(teacher_net, decoder_net, dataset, num_blocks, k, device, writer):
    teacher_net = teacher_net.to(device)
    decoder_net = decoder_net.to(device)
    criterion = ChamferDistanceLoss().to(device)
    knn_graph = KNNGraph().to(device)
    optimizer = torch.optim.Adam(list(teacher_net.parameters()) + list(decoder_net.parameters()), lr=1e-3, weight_decay=1e-6)
    s = 0.015
    scaling_factor = 1 / s
    dataset_size = len(dataset)
    for epoch in range(250):
        print(f'EPOCH: {epoch}')
        epoch_loss = 0
        for step, data in tqdm(enumerate(dataset), desc='Training'):
            point_cloud = data[0].to(device) * scaling_factor
            knn = knn_graph(point_cloud, k)

            descriptors = teacher_net(point_cloud, knn)
            sampled_indices = torch.randint(0, len(point_cloud), (NUM_FEATURE_SAMPLES,))

            total_loss = []
            reconstructed_points = decoder_net(descriptors[sampled_indices])
            for i, pts in zip(sampled_indices, reconstructed_points):
                # 2 LFA per resnet block
                rf_indices = get_rf(knn, i, num_blocks * 2)
                rf = point_cloud[rf_indices]
                rf_centered = rf - torch.mean(rf, dim=0)
                loss = criterion(pts.reshape(-1, 3), rf_centered)
                total_loss.append(loss)

            total_loss = sum(total_loss) / NUM_FEATURE_SAMPLES
            epoch_loss += total_loss.detach()

            if writer is not None:
                writer.add_scalar('Loss/Total', total_loss, epoch * dataset_size + step)

            optimizer.zero_grad()
            total_loss.backward()
            # print(teacher_net.res_block.lfa2.mlp.mlp.weight.grad)
            optimizer.step()

        print(f'Epoch Loss: {epoch_loss / 500}')
        if writer is not None:
            writer.add_scalar('Epoch Loss/Total', epoch_loss / 500, epoch)

        if epoch % 25 == 0 and epoch is not 0:
            torch.save(teacher_net.state_dict(), '/baldig/chemistry/2023_rp/Chemformer/pivot/models/teacher.pt')
            torch.save(decoder_net.state_dict(), '/baldig/chemistry/2023_rp/Chemformer/pivot/models/decoder.pt')

    return teacher_net, decoder_net


if __name__ == '__main__':
    d_model = 64
    k = 8
    num_blocks = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter()

    t_net = TeacherNetwork(d_model, k, num_blocks, device=device).to(device)
    decoder = DecoderNetwork(d_model, 128).to(device)

    train_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH_16K / 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    test_dataset = ModelNetDataset(root_dir=SYNTHETIC_DATA_PATH_16K / 'test')
    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    t, d = pretrain_teacher_network(
        t_net,
        decoder,
        train_dataloader,
        num_blocks,
        k,
        device,
        writer
    )