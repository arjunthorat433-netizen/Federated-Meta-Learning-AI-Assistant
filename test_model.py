import torch
from federated_app.task import ClientModel, load_test_data

model = ClientModel()

state_dict = torch.load(
    "final_model.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("\nMODEL LOADED SUCCESSFULLY!")

for client_id in (1, 2, 3):
    X, y = load_test_data(client_id)

    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)

    accuracy = (predictions == y).float().mean().item()

    print(f"Client {client_id} Independent Test Accuracy: {accuracy * 100:.2f}%")

print("\nFINAL MODEL TESTING COMPLETED!")
