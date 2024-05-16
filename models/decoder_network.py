import torch
import torch.nn as nn


class DecoderNetwork(nn.Module):
    def __init__(self, d_model, num_points=1024):
        super().__init__()
        self.mlp = nn.Linear(d_model, 3 * num_points)

    def forward(self, inputs):
        # inputs: [d]
        # output: [3m]
        return self.mlp(inputs)
