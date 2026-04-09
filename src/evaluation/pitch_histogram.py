import numpy as np


def pitch_histogram_similarity(real_hist: np.ndarray, gen_hist: np.ndarray) -> float:
    return float(np.abs(real_hist - gen_hist).sum())
