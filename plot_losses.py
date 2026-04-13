# file: plot_losses.py

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_loss_series(data):
    if isinstance(data, dict) and "values" in data:
        return [float(x) for x in data["values"]]
    if isinstance(data, list):
        return [float(x) for x in data]
    raise ValueError(f"Unsupported JSON format: {type(data)}")


ae_train = extract_loss_series(load_json("outputs/plots/ae_train_loss.json"))
ae_valid = extract_loss_series(load_json("outputs/plots/ae_valid_loss.json"))
vae_loss = extract_loss_series(load_json("outputs/plots/vae_loss.json"))

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ae_train) + 1), ae_train, label="AE Train Loss")
plt.plot(range(1, len(ae_valid) + 1), ae_valid, label="AE Valid Loss")
plt.plot(range(1, len(vae_loss) + 1), vae_loss, marker="o", label="VAE Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/loss_curves.png")
plt.show()