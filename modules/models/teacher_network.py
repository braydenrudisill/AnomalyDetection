import torch
import torch.nn as nn

from modules.models import KNNGraph


class TeacherNetwork(nn.Module):
    def __init__(self, d_model, k, n_res_blocks, device):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.n_res_blocks = n_res_blocks
        self.device = device

        assert self.n_res_blocks > 0, f"Number of residual blocks must be greater than 0 but was {n_res_blocks}"

        self.residual_mlp = SharedMLP(d_model, d_model)
        self.res_block = ResBlock(d_model, k)
        self.knn_graph = KNNGraph()

    def forward(self, inputs, knn=None):
        if knn is None:
            knn = self.knn_graph(inputs, self.k).to(self.device)

        diffs = inputs.unsqueeze(1).expand(-1, self.k, -1) - inputs[knn]
        norms = torch.norm(diffs, dim=-1, keepdim=True)
        geometric_features = torch.cat([diffs, norms], dim=-1)

        f0 = torch.zeros([inputs.size(0), self.d_model]).to(self.device)

        for _ in range(self.n_res_blocks):
            x = self.res_block(f0, geometric_features, knn)
            x += self.residual_mlp(f0)
            f0 = x

        return x


class ResBlock(nn.Module):
    def __init__(self, d_model, k):
        super().__init__()
        self.d_model = d_model
        self.mlp1 = SharedMLP(d_model, d_model // 4)
        self.lfa = LocalFeatureAggregation(d_model // 4, k)
        self.lfa2 = LocalFeatureAggregation(d_model // 2, k)
        self.mlp2 = SharedMLP(d_model, d_model)

    def forward(self, inputs, geometric_features, knn):
        x = self.mlp1(inputs)
        x = self.lfa(x, geometric_features, knn)
        x = self.lfa2(x, geometric_features, knn)
        x = self.mlp2(x)
        return x


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, k):
        super().__init__()
        self.k = k
        self.mlp = SharedMLP(4, d_in)

    def forward(self, inputs, geometric_features, knn):
        # Geometric features: [num_pts, k, 4]
        # Input features: [batch, num_pts, d_in]
        x = self.mlp(geometric_features)
        x = torch.cat((x, inputs[knn]), dim=2)
        x = torch.mean(x, dim=1)
        return x


class SharedMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.mlp = nn.Linear(d_in, d_out)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        x = self.mlp(inputs)
        x = self.relu(x)
        return x
