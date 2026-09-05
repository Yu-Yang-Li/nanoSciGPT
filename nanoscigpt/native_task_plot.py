"""Plot asset for task-specific native v1 experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


labels = {"run_0": "Baseline"}
figure, axes = plt.subplots()
for run, label in labels.items():
    values = json.loads((Path(run) / "training_losses.json").read_text())
    axes.plot(range(1, len(values) + 1), values, label=label)
axes.set(xlabel="Training step", ylabel="Training loss")
axes.legend()
figure.tight_layout()
figure.savefig("task_training.png", dpi=150)
plt.close(figure)
