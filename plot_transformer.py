# file: plot_transformer.py

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_values(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "values" in data:
        return [float(x) for x in data["values"]]
    if isinstance(data, list):
        return [float(x) for x in data]
    raise ValueError(f"Unsupported JSON format in {path}")


loss = load_values("outputs/plots/transformer_loss.json")
perplexity = load_values("outputs/plots/transformer_perplexity.json")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss) + 1), loss, label="Transformer Loss")
plt.plot(range(1, len(perplexity) + 1), perplexity, label="Transformer Perplexity")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Transformer Training Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/transformer_curves.png")
plt.show()