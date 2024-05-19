import torch
from torch import nn as nn


class KNNGraph(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, k):
        distances, indices = torch.topk(torch.cdist(x, x), k + 1, largest=False)
        # Discard the first column which represents the distance of each point to itself
        return indices[:, 1:]


if __name__ == '__main__':
    knn_graph = KNNGraph()
    points = torch.tensor([[0,0,0], [1,1,1], [1,1.1,1]])
    print(knn_graph(points, 2).shape)
