import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.shortcut = nn.Linear(input_dim, 128)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.fc4(out).squeeze(-1)
        return out

class ResidualAttentionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.se = SEBlock(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.shortcut = nn.Linear(input_dim, 128)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.se(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.fc4(out).squeeze(-1)
        return out

class EnsembleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model1 = ResidualMLP(input_dim)
        self.model2 = ResidualAttentionMLP(input_dim)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return (out1 + out2) / 2
