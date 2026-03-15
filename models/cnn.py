import torch
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(

            nn.Linear(num_features * 32, 128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, x, mask):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = x.flatten(1)

        return self.fc(x)