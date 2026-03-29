import torch.nn as nn


class MnistModel(nn.Module):
    def __init__(self, conv: bool = False):
        super().__init__()

        if conv:
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(p=0.4),
                nn.Linear(16 * 7 * 7, 10),
            )
        else:
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(28 * 28, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
            )

    def forward(self, x):
        return self.net(x)
