import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pretty_midi

from .tokenizer import MusicTokenizer


def parse_midi_to_notes(midi_path: Path) -> List[Tuple[int, int, float]]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    # Tempo estimation is relatively expensive; compute it once per file.
    tempo_scale = pm.estimate_tempo() / 60.0
    notes: List[Tuple[int, int, float]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            duration_sec = max(1e-3, n.end - n.start)
            duration_beats = duration_sec * tempo_scale
            notes.append((n.pitch, n.velocity, float(duration_beats)))
    notes.sort(key=lambda x: x[2])
    return notes


def chunk_sequence(seq: List[int], seq_len: int, step: int) -> List[List[int]]:
    if len(seq) < seq_len:
        return []
    return [seq[i : i + seq_len] for i in range(0, len(seq) - seq_len + 1, step)]


def build_dataset(input_dir: Path, output_dir: Path, seq_len: int, step: int) -> None:
    tokenizer = MusicTokenizer()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    genres = []
    genre_to_idx = {}

    for genre_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        genre = genre_dir.name
        genre_idx = genre_to_idx.setdefault(genre, len(genre_to_idx))

        # One recursive walk is cheaper than scanning twice for each extension.
        midi_files = [
            p for p in genre_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".mid", ".midi"}
        ]
        for midi_file in midi_files:
            try:
                notes = parse_midi_to_notes(midi_file)
                if len(notes) < 32:
                    continue
                token_ids = tokenizer.encode_events(notes)
                chunks = chunk_sequence(token_ids, seq_len, step)
                if not chunks:
                    continue
                sequences.extend(chunks)
                genres.extend([genre_idx] * len(chunks))
            except Exception:
                continue

    if not sequences:
        raise RuntimeError("No valid MIDI sequences found. Check data/raw_midi directory.")

    x = np.array(sequences, dtype=np.int64)
    y = np.array(genres, dtype=np.int64)

    np.savez(output_dir / "sequences.npz", x=x, y=y)
    tokenizer.save_vocab(output_dir / "vocab.json")

    meta = {
        "num_sequences": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "num_genres": len(genre_to_idx),
        "genres": genre_to_idx,
        "vocab_size": tokenizer.vocab_size,
    }
    (output_dir / "metadata.txt").write_text(str(meta), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw MIDI root")
    parser.add_argument("--output", type=str, required=True, help="Path to processed output")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--step", type=int, default=16)
    args = parser.parse_args()

    build_dataset(Path(args.input), Path(args.output), args.seq_len, args.step)
