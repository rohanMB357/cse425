from typing import List, Tuple

import numpy as np


def notes_to_piano_roll(
    notes: List[Tuple[int, int, float]],
    steps_per_beat: int = 4,
    pitch_min: int = 21,
    pitch_max: int = 108,
) -> np.ndarray:
    pitch_range = pitch_max - pitch_min + 1
    total_steps = max(1, int(sum(d for _, _, d in notes) * steps_per_beat))
    roll = np.zeros((total_steps, pitch_range), dtype=np.float32)

    cursor = 0
    for pitch, _velocity, duration in notes:
        p = pitch - pitch_min
        if 0 <= p < pitch_range:
            dur_steps = max(1, int(round(duration * steps_per_beat)))
            end = min(total_steps, cursor + dur_steps)
            roll[cursor:end, p] = 1.0
            cursor = end
    return roll


def roll_to_pitch_histogram(roll: np.ndarray, pitch_min: int = 21) -> np.ndarray:
    pitch_usage = roll.sum(axis=0)
    histogram_12 = np.zeros(12, dtype=np.float32)
    for idx, val in enumerate(pitch_usage):
        midi_pitch = pitch_min + idx
        histogram_12[midi_pitch % 12] += val
    if histogram_12.sum() > 0:
        histogram_12 = histogram_12 / histogram_12.sum()
    return histogram_12
