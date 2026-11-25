import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_layers=1, hidden=64, dropout=0.0):
        super().__init__()
        layers = []
        in_features = 64
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_features, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, 10))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
