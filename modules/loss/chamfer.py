import torch


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b) -> torch.Tensor:
        dists = torch.cdist(a, b)
        d_ab, _ = torch.min(dists, dim=0)
        d_ba, _ = torch.min(dists, dim=1)
        return torch.mean(torch.cat([d_ab, d_ba]))
