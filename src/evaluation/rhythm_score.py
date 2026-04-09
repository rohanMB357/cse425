import numpy as np


def rhythm_diversity(durations: np.ndarray) -> float:
    if durations.size == 0:
        return 0.0
    return float(len(np.unique(durations)) / len(durations))
