# file: baseline_metrics.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import evaluate, markov_baseline, random_baseline


def save_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    data = np.load("data/processed_small_fast/sequences.npz")
    real = data["x"]

    vocab_size = int(real.max()) + 1
    seq_len = int(real.shape[1])

    random_gen = random_baseline(vocab_size=vocab_size, seq_len=seq_len, n=100)
    markov_gen = markov_baseline(train_tokens=real, seq_len=seq_len, n=100)

    random_metrics = evaluate(real, random_gen)
    markov_metrics = evaluate(real, markov_gen)

    save_json("outputs/plots/random_metrics.json", random_metrics)
    save_json("outputs/plots/markov_metrics.json", markov_metrics)

    print("Random baseline:")
    print(json.dumps(random_metrics, indent=2))
    print()
    print("Markov baseline:")
    print(json.dumps(markov_metrics, indent=2))


if __name__ == "__main__":
    main()