import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from pathlib import Path

from meta_learning.preference_model import PreferenceMLP
from meta_learning.maml import MAML, generate_user_tasks


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("MAML Training Started")
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

torch.manual_seed(42)

model = PreferenceMLP()

maml = MAML(
    model=model,
    learning_rate=0.01,
    meta_learning_rate=0.001,
    inner_steps=2,
    device=device,
)

tasks = generate_user_tasks(
    number_of_users=20,
    support_size=16,
    query_size=16,
)

maml.meta_train(tasks, epochs=100)

save_path = MODEL_DIR / "preference_maml.pt"
maml.save(save_path)

print("\nMAML Training Completed!")
print("Model saved:", save_path)
