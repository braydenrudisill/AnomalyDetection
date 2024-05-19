import torch.nn as nn


class DecoderNetwork(nn.Module):
    def __init__(self, d_model, num_points=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.05),
            nn.Linear(128, num_points*3),
        )

    def forward(self, inputs):
        # inputs: [batch_size, d]
        # output: [batch_size, 3m]
        x = self.mlp(inputs)
        return x
