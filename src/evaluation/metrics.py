from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.pitch_histogram import pitch_histogram_similarity
from src.evaluation.rhythm_score import rhythm_diversity


def repetition_ratio(tokens: np.ndarray, ngram: int = 4) -> float:
    if tokens.size < ngram:
        return 0.0
    patterns = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    unique = len(set(patterns))
    return float((len(patterns) - unique) / max(1, len(patterns)))


def random_baseline(vocab_size: int, seq_len: int, n: int = 100) -> np.ndarray:
    return np.random.randint(0, vocab_size, size=(n, seq_len), dtype=np.int64)


def markov_baseline(train_tokens: np.ndarray, seq_len: int, n: int = 100) -> np.ndarray:
    transitions: dict[int, list[int]] = {}
    flat = train_tokens.flatten()
    for i in range(len(flat) - 1):
        transitions.setdefault(int(flat[i]), []).append(int(flat[i + 1]))

    starts = list(transitions.keys())
    samples = []
    for _ in range(n):
        cur = int(np.random.choice(starts))
        seq = [cur]
        for _ in range(seq_len - 1):
            nxts = transitions.get(cur, starts)
            cur = int(np.random.choice(nxts))
            seq.append(cur)
        samples.append(seq)
    return np.array(samples, dtype=np.int64)


def evaluate(real: np.ndarray, generated: np.ndarray) -> dict:
    real_flat = real.flatten()
    gen_flat = generated.flatten()

    real_hist = np.bincount(real_flat, minlength=max(real_flat.max(), gen_flat.max()) + 1).astype(np.float64)
    gen_hist = np.bincount(gen_flat, minlength=real_hist.shape[0]).astype(np.float64)
    if real_hist.sum() > 0:
        real_hist /= real_hist.sum()
    if gen_hist.sum() > 0:
        gen_hist /= gen_hist.sum()

    duration_proxy_real = real_flat % 12
    duration_proxy_gen = gen_flat % 12

    return {
        "pitch_histogram_similarity": pitch_histogram_similarity(real_hist[:12], gen_hist[:12]),
        "rhythm_diversity_real": rhythm_diversity(duration_proxy_real),
        "rhythm_diversity_gen": rhythm_diversity(duration_proxy_gen),
        "repetition_ratio_real": repetition_ratio(real_flat),
        "repetition_ratio_gen": repetition_ratio(gen_flat),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--generated", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    real = np.load(args.real)["x"]
    gen_payload = np.load(args.generated)
    generated = gen_payload["x"] if "x" in gen_payload else gen_payload[list(gen_payload.keys())[0]]

    result = evaluate(real, generated)
    out_path = Path(args.out) / "metrics_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
