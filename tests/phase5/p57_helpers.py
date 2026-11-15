"""Shared helpers for the P5.7 tests (a uniquely named module: ``conftest`` names collide across tests/phase4 and tests/phase5)."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_tiny_training_config(path, **over):
    """A development-scale training configuration derived from the default one (2 epochs, batch 8, single CPU thread)."""
    base = json.loads((REPO_ROOT / "configs" / "phase5" / "training_default.json").read_text(encoding="utf-8"))
    base.update(epochs=2, batch_size=8, log_every_steps=1, checkpoint_every_epochs=1, scaler_batch_size=8, torch_threads=1)
    base.update(over)
    Path(path).write_text(json.dumps(base), encoding="utf-8")
    return Path(path)
