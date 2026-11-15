
import torch
import torch.nn as nn
import torch.optim as optim



class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)



def load_data(client_id):
    torch.manual_seed(client_id)

    X = torch.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).long()

    return X, y
    # Independent Test Data
def load_test_data(client_id):
    generator = torch.Generator().manual_seed(1000 + client_id)

    X = torch.randn(200, 4, generator=generator)
    y = (X[:, 0] + X[:, 1] > 0).long()

    return X, y



def train(model, client_id, epochs=5):
    X, y = load_data(client_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

    return loss.item()



def test(model, client_id):
    X, y = load_test_data(client_id)

    model.eval()

    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean().item()

    return accuracy