
import torch
from pathlib import Path

from meta_learning.preference_model import PreferenceMLP
from meta_learning.maml import MAML

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "preference_maml.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PreferenceMLP()

maml = MAML(
    model=model,
    learning_rate=0.01,
    inner_steps=5,
    device=device,
)

maml.load(MODEL_PATH)
model.eval()

# New user's response preferences.
# Features:
# concise, detailed, examples, formal,
# friendly, technical, step_by_step, bullet_points

support_x = torch.tensor([
    [0.9, 0.1, 0.2, 0.2, 0.9, 0.3, 0.8, 0.9],
    [0.8, 0.2, 0.1, 0.2, 0.8, 0.2, 0.9, 0.8],
    [0.2, 0.9, 0.9, 0.3, 0.8, 0.4, 0.9, 0.7],
    [0.9, 0.1, 0.1, 0.8, 0.2, 0.9, 0.2, 0.2],
], dtype=torch.float32, device=device)

# 1 = liked, 0 = disliked
support_y = torch.tensor(
    [1, 1, 1, 0], dtype=torch.float32, device=device
)

query_x = torch.tensor([
    [0.9, 0.1, 0.2, 0.2, 0.9, 0.3, 0.8, 0.9],
    [0.1, 0.9, 0.9, 0.8, 0.2, 0.9, 0.2, 0.2],
], dtype=torch.float32, device=device)

initial_params = dict(model.named_parameters())

with torch.no_grad():
    before = torch.sigmoid(
        maml.forward(query_x, initial_params)
    )

adapted_params = maml.adapt(
    support_x,
    support_y,
    params=initial_params,
    first_order=True,
)

with torch.no_grad():
    after = torch.sigmoid(
        maml.forward(query_x, adapted_params)
    )

print("\nNEW USER PERSONALIZATION")
print("------------------------")

for i in range(len(query_x)):
    print(f"Candidate {i + 1}")
    print(f"Before adaptation: {before[i].item():.4f}")
    print(f"After adaptation:  {after[i].item():.4f}")

print("\nPersonalization test completed!")
