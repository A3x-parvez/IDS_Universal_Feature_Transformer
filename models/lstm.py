import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 2)

    def forward(self, x, mask):

        x = x.unsqueeze(-1)

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.fc(out)