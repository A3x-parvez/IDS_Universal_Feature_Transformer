import torch
import torch.nn as nn


class ANNModel(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x, mask):

        return self.model(x)