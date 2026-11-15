
import torch
from pathlib import Path

from meta_learning.preference_model import PreferenceMLP
from meta_learning.maml import MAML

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "preference_maml.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = PreferenceMLP()
maml = MAML(
    model,
    learning_rate=0.01,
    inner_steps=5,
    device=DEVICE,
)

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"MAML checkpoint not found: {CHECKPOINT}"
    )

maml.load(CHECKPOINT)
model.eval()


def personalize(feedback, candidates):
    """
    feedback: list of {"features": [8 values], "liked": 0 or 1}
    candidates: list of [8 feature values]
    """
    if not feedback:
        raise ValueError("At least one feedback example is required.")

    support_x = torch.tensor(
        [item["features"] for item in feedback],
        dtype=torch.float32,
        device=DEVICE,
    )
    support_y = torch.tensor(
        [item["liked"] for item in feedback],
        dtype=torch.float32,
        device=DEVICE,
    )
    query_x = torch.tensor(
        candidates,
        dtype=torch.float32,
        device=DEVICE,
    )

    if support_x.ndim != 2 or support_x.shape[1] != 8:
        raise ValueError("Each feedback example needs 8 features.")

    if query_x.ndim != 2 or query_x.shape[1] != 8:
        raise ValueError("Each candidate needs 8 features.")

    if not torch.all((support_y == 0) | (support_y == 1)):
        raise ValueError("Feedback labels must be 0 or 1.")

    params = dict(model.named_parameters())

    adapted = maml.adapt(
        support_x,
        support_y,
        params=params,
        first_order=True,
    )

    with torch.no_grad():
        scores = torch.sigmoid(maml.forward(query_x, adapted))

    return scores.cpu().tolist()
