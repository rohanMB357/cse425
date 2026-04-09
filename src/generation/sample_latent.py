from __future__ import annotations

import numpy as np


def sample_gaussian_latent(batch: int, latent_dim: int) -> np.ndarray:
    return np.random.randn(batch, latent_dim).astype(np.float32)


def sample_markov(transitions: dict[int, list[int]], start: int, length: int) -> list[int]:
    seq = [start]
    cur = start
    keys = list(transitions.keys()) or [start]
    for _ in range(length - 1):
        nxts = transitions.get(cur, keys)
        cur = int(np.random.choice(nxts))
        seq.append(cur)
    return seq
