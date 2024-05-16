import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from teacher_network import knn_graph


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(LocalFeatureAggregation, self).__init__()
        self.k = k
        self.mlp = SharedMLP(in_channels * 2, out_channels)

    def forward(self, x, pos):
        # x: (batch_size, num_points, in_channels)
        # pos: (batch_size, num_points, 3)
        batch_size, num_points, _ = x.size()
        edge_index = knn_graph(pos, k=self.k, batch=batch_size).to(x.device)

        row, col = edge_index
        diff = pos[row] - pos[col]  # (num_edges, 3)
        diff = torch.cat([diff, torch.norm(diff, dim=1, keepdim=True)], dim=-1)  # (num_edges, 4)

        edge_attr = self.mlp(diff.unsqueeze(0)).squeeze(0)  # (num_edges, out_channels)
        edge_attr = F.leaky_relu(edge_attr, 0.1)

        # Aggregate features
        out = pyg_nn.global_mean_pool(edge_attr, row, batch_size * num_points)
        out = out.view(batch_size, num_points, -1)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(ResidualBlock, self).__init__()
        self.lfa1 = LocalFeatureAggregation(in_channels, out_channels, k)
        self.lfa2 = LocalFeatureAggregation(out_channels, out_channels, k)
        self.mlp = SharedMLP(in_channels, out_channels)

    def forward(self, x, pos):
        identity = self.mlp(x.transpose(1, 2)).transpose(1, 2)
        out = self.lfa1(x, pos)
        out = self.lfa2(out, pos)
        out += identity
        return F.relu(out)


class TeacherNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, k, num_blocks):
        super(TeacherNetwork, self).__init__()
        self.shared_mlp = SharedMLP(in_channels, out_channels)
        self.blocks = nn.ModuleList([ResidualBlock(out_channels, out_channels, k) for _ in range(num_blocks)])
        self.final_mlp = SharedMLP(out_channels, out_channels)

    def forward(self, x, pos):
        x = self.shared_mlp(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x, pos)
        x = self.final_mlp(x.transpose(1, 2)).transpose(1, 2)
        return x


class DecoderNetwork(nn.Module):
    def __init__(self, in_channels, num_points):
        super(DecoderNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_points * 3)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(x.size(0), -1, 3)
        return x

def pretrain_teacher_network(teacher_net, decoder_net, point_clouds, num_points, k, device):
    teacher_net = teacher_net.to(device)
    decoder_net = decoder_net.to(device)
    criterion = ChamferLoss().to(device)
    optimizer = torch.optim.Adam(list(teacher_net.parameters()) + list(decoder_net.parameters()), lr=1e-3)

    for point_cloud in point_clouds:
        point_cloud = point_cloud.to(device)
        pos = point_cloud[:, :3]
        x = torch.zeros(point_cloud.size(0), point_cloud.size(1), in_channels).to(device)

        descriptors = teacher_net(x, pos)
        sampled_indices = torch.randint(0, point_cloud.size(1), (num_points,))
        sampled_descriptors = descriptors[:, sampled_indices, :]

        reconstructed_points = decoder_net(sampled_descriptors)

        with torch.no_grad():
            receptive_fields = []
            for idx in sampled_indices:
                rf_indices = knn_graph(pos[:, idx].unsqueeze(1), k=k, batch=pos.size(0)).to(device)
                rf_points = pos[:, rf_indices[1]].view(pos.size(0), k, 3)
                mean_rf_points = rf_points.mean(dim=1, keepdim=True)
                receptive_fields.append(rf_points - mean_rf_points)
            receptive_fields = torch.stack(receptive_fields, dim=1)

        loss = criterion(reconstructed_points, receptive_fields)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return teacher_net, decoder_net


# Example usage
in_channels = 3
out_channels = 64
k = 20
num_blocks = 4
num_points = 1024

teacher_net = TeacherNetwork(in_channels, out_channels, k, num_blocks)
decoder_net = DecoderNetwork(out_channels, num_points)

point_clouds = [torch.rand(1, num_points, 3)]  # Example point cloud data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_teacher_net, pretrained_decoder_net = pretrain_teacher_network(teacher_net, decoder_net, point_clouds,
                                                                          num_points, k, device)
