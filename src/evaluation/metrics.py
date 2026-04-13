# file: src/evaluation/metrics.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.pitch_histogram import pitch_histogram_similarity
from src.evaluation.rhythm_score import rhythm_diversity
from src.preprocessing.midi_parser import parse_midi_to_notes
from src.preprocessing.tokenizer import MusicTokenizer


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


def load_npz_tokens(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "x" in payload:
        return payload["x"]
    first_key = list(payload.keys())[0]
    return payload[first_key]


def load_midi_folder_tokens(path: Path, tokenizer: MusicTokenizer, seq_len: int = 64) -> np.ndarray:
    midi_files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".mid", ".midi"}]
    sequences: list[np.ndarray] = []

    for midi_file in midi_files:
        try:
            notes = parse_midi_to_notes(midi_file)
            if not notes:
                continue
            token_ids = tokenizer.encode_events(notes)
            token_ids = np.array(token_ids[:seq_len], dtype=np.int64)

            if token_ids.size < seq_len:
                pad_id = tokenizer.stoi["<PAD>"]
                token_ids = np.pad(token_ids, (0, seq_len - token_ids.size), constant_values=pad_id)

            sequences.append(token_ids)
        except Exception:
            continue

    if not sequences:
        raise RuntimeError(f"No valid MIDI files found in: {path}")

    return np.stack(sequences, axis=0)


def load_tokens(path_str: str, tokenizer: MusicTokenizer, seq_len: int = 64) -> np.ndarray:
    path = Path(path_str)
    if path.is_file() and path.suffix.lower() == ".npz":
        return load_npz_tokens(path)
    if path.is_dir():
        return load_midi_folder_tokens(path, tokenizer, seq_len=seq_len)
    raise FileNotFoundError(f"Unsupported input path: {path}")


def evaluate(real: np.ndarray, generated: np.ndarray) -> dict:
    real_flat = real.flatten()
    gen_flat = generated.flatten()

    max_token = int(max(real_flat.max(initial=0), gen_flat.max(initial=0))) + 1
    real_hist = np.bincount(real_flat, minlength=max_token).astype(np.float64)
    gen_hist = np.bincount(gen_flat, minlength=max_token).astype(np.float64)

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
    parser.add_argument("--out", type=str, default="outputs/plots/metrics_summary.json")
    parser.add_argument("--vocab", type=str, default="data/processed_small_fast/vocab.json")
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    tokenizer = MusicTokenizer.from_vocab(args.vocab)

    real = load_tokens(args.real, tokenizer, seq_len=args.seq_len)
    generated = load_tokens(args.generated, tokenizer, seq_len=args.seq_len)

    result = evaluate(real, generated)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))