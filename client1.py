import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)



def load_data():
    torch.manual_seed(1)

    X = torch.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()

    return X, y



def train_client():
    print("CLIENT 1 TRAINING STARTED")

    X, y = load_data()

    model = ClientModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("CLIENT 1 TRAINING COMPLETED")

    return model


if __name__ == "__main__":
    train_client()