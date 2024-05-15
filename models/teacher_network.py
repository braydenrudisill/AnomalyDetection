import torch.nn as nn


class TeacherNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.model = ResNet(
            nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
        )

    def forward(self, inputs):
        logits = self.model(inputs)
        return logits


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
