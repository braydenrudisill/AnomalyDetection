import torch
import torch.nn as nn


class TeacherNetwork(nn.Module):
    def __init__(self, d_model, k, n_points, n_res_blocks, device):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.n_points = n_points
        self.n_res_blocks = n_res_blocks
        self.device = device

        assert self.n_res_blocks > 0, f"Number of residual blocks must be greater than 0 but was {n_res_blocks}"

        self.residual_mlp = nn.Linear(d_model, d_model)
        self.res_block = ResBlock(d_model, n_points, k)

    def forward(self, inputs, knn=None):
        if knn is None:
            knn = knn_graph(inputs, self.k).to(self.device)

        geometric_features = []
        for p_idx in range(self.n_points):
            features = []
            for neighbor_idx in knn[p_idx]:
                diff = inputs[p_idx] - inputs[neighbor_idx]
                norm = torch.norm(diff)
                features.append(torch.cat([diff, torch.tensor([norm], device=self.device)]))
            geometric_features.append(torch.stack(features))
        geometric_features = torch.stack(geometric_features)

        f0 = torch.zeros([self.n_points, self.d_model]).to(self.device)
        for _ in range(self.n_res_blocks):
            x = self.res_block(f0, geometric_features, knn)
            x += self.residual_mlp(f0)
            f0 = x

        return x


class ResBlock(nn.Module):
    def __init__(self, d_model, n_points, k):
        super().__init__()
        self.d_model = d_model
        self.mlp1 = nn.Linear(d_model, d_model // 4)
        self.lfa1 = LocalFeatureAggregation(d_model // 4, n_points, k)
        self.lfa2 = LocalFeatureAggregation(d_model // 2, n_points, k)
        self.mlp2 = nn.Linear(d_model, d_model)

    def forward(self, inputs, geometric_features, knn):
        x = self.mlp1(inputs)
        x = self.lfa1(x, geometric_features, knn)
        x = self.lfa2(x, geometric_features, knn)
        x = self.mlp2(x)
        x += inputs
        return x


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, n_points, k):
        super().__init__()
        self.d_in = d_in
        self.n_points = n_points
        self.k = k

        self.mlp = nn.Linear(4, d_in)
        self.pool = nn.AvgPool3d(kernel_size=k)

    def forward(self, inputs, geometric_features, knn):
        # Geometric features: [batch, num_pts, k, 4]
        # Input features: [batch, num_pts, d_in]
        x = self.mlp(geometric_features)
        x = torch.cat((x, inputs[knn]), dim=2)
        x = torch.mean(x, dim=1)
        return x


class KNNGraph(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, inputs):
        return knn_graph(inputs, self.k)


def knn_graph(x, k):
    dist = torch.sort(torch.cdist(x, x), dim=1)
    indices = dist.indices[:, 1: k + 1]
    return indices


if __name__ == '__main__':
    points = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 1, 1.1]], dtype=torch.float)
    print(knn_graph(points, 2))
