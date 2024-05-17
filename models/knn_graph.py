import torch
from torch import nn as nn


class KNNGraph(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, k):
        dist = torch.sort(torch.cdist(x, x), dim=1)
        indices = dist.indices[:, 1: k + 1]
        return indices
