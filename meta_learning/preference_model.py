import torch
import torch.nn as nn
import torch.nn.functional as F


PREFERENCE_FEATURES = [
    "concise",
    "detailed",
    "examples",
    "formal",
    "friendly",
    "technical",
    "step_by_step",
    "bullet_points",
]


class PreferenceMLP(nn.Module):
    def __init__(self, input_size=8, hidden_size=32):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)