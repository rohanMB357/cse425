from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import pretty_midi


def tokens_to_midi_notes(
    notes: Sequence[Tuple[int, int, float]],
    out_path: str | Path,
    tempo: float = 120.0,
) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=0)

    t = 0.0
    sec_per_beat = 60.0 / tempo
    for pitch, velocity, dur_beats in notes:
        dur_sec = max(0.05, dur_beats * sec_per_beat)
        n = pretty_midi.Note(
            velocity=int(max(1, min(127, velocity))),
            pitch=int(max(21, min(108, pitch))),
            start=t,
            end=t + dur_sec,
        )
        inst.notes.append(n)
        t += dur_sec

    pm.instruments.append(inst)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
