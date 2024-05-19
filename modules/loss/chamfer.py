import torch


class ChamferDistance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b) -> torch.Tensor:
        diff = a[:, None, :] - b[None, :, :]
        dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))

        min_dist, _ = torch.min(dist_matrix, dim=1)
        min_dist2, _ = torch.min(dist_matrix, dim=0)

        chamfer_dist = torch.mean(torch.concat([min_dist, min_dist2]))
        return chamfer_dist
