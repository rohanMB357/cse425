# file: src/generation/generate_music.py

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch

from src.generation.midi_export import tokens_to_midi_notes
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import MusicTokenizer


def load_dataset(npz_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    x = torch.tensor(data["x"], dtype=torch.long)
    y = torch.tensor(data["y"], dtype=torch.long)
    return x, y


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_token_ids_to_notes(
    tokenizer: MusicTokenizer,
    token_ids: list[int],
) -> list[tuple[int, int, float]]:
    notes: list[tuple[int, int, float]] = []

    current_pitch: int | None = None
    current_velocity: int | None = None
    current_duration: float | None = None

    for token_id in token_ids:
        token = tokenizer.itos.get(int(token_id), "<PAD>")

        if token in tokenizer.special_tokens:
            continue

        if token.startswith("PITCH_"):
            if (
                current_pitch is not None
                and current_velocity is not None
                and current_duration is not None
            ):
                notes.append((current_pitch, current_velocity, current_duration))
            current_pitch = int(token.split("_")[1])
            current_velocity = None
            current_duration = None
            continue

        if token.startswith("VEL_") and current_pitch is not None:
            vel_bin = int(token.split("_")[1])
            current_velocity = int((vel_bin + 0.5) / tokenizer.cfg.velocity_bins * 127)
            continue

        if token.startswith("DUR_") and current_pitch is not None:
            current_duration = float(token.split("_")[1])
            if current_velocity is None:
                current_velocity = 80
            notes.append((current_pitch, current_velocity, current_duration))
            current_pitch = None
            current_velocity = None
            current_duration = None

    return notes


def save_token_sequence_as_midi(
    tokenizer: MusicTokenizer,
    token_ids: list[int],
    out_path: Path,
    tempo: float = 120.0,
) -> bool:
    notes = decode_token_ids_to_notes(tokenizer, token_ids)
    if not notes:
        print(f"Skipped empty/invalid sequence: {out_path}")
        return False
    tokens_to_midi_notes(notes, out_path, tempo=tempo)
    return True


def call_model_generate(
    model: MusicTransformer,
    seed: torch.Tensor,
    genre_id: int,
    new_tokens: int,
    temperature: float,
) -> torch.Tensor:
    sig = inspect.signature(model.generate)
    params = list(sig.parameters.keys())

    if "seed" in params:
        kwargs: dict[str, object] = {"seed": seed}
    elif "x" in params:
        kwargs = {"x": seed}
    elif "prompt" in params:
        kwargs = {"prompt": seed}
    else:
        kwargs = {}
        positional_seed = True

    if "genre_id" in params:
        kwargs["genre_id"] = genre_id
    elif "genre" in params:
        kwargs["genre"] = genre_id

    if "max_new_tokens" in params:
        kwargs["max_new_tokens"] = new_tokens
    elif "new_tokens" in params:
        kwargs["new_tokens"] = new_tokens
    elif "length" in params:
        kwargs["length"] = new_tokens

    if "temperature" in params:
        kwargs["temperature"] = temperature

    if "seed" not in params and "x" not in params and "prompt" not in params:
        return model.generate(seed, **kwargs)

    return model.generate(**kwargs)


@torch.no_grad()
def generate_sequences(
    model: MusicTransformer,
    x: torch.Tensor,
    n_samples: int,
    seed_len: int,
    new_tokens: int,
    genre_id: int,
    temperature: float,
    device: torch.device,
) -> list[list[int]]:
    model.eval()

    n_samples = min(n_samples, x.size(0))
    seeds = x[:n_samples, :seed_len].to(device)

    generated: list[list[int]] = []
    for i in range(seeds.size(0)):
        seed = seeds[i : i + 1]
        seq = call_model_generate(
            model=model,
            seed=seed,
            genre_id=genre_id,
            new_tokens=new_tokens,
            temperature=temperature,
        )
        generated.append(seq.squeeze(0).tolist())

    return generated


def main(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    tokenizer = MusicTokenizer.from_vocab(args.vocab)
    x, y = load_dataset(args.data)

    vocab_size = tokenizer.vocab_size
    num_genres = int(y.max().item()) + 1

    model = MusicTransformer(vocab_size=vocab_size, num_genres=num_genres).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_sequences(
        model=model,
        x=x,
        n_samples=args.n_samples,
        seed_len=args.seed_len,
        new_tokens=args.new_tokens,
        genre_id=args.genre_id,
        temperature=args.temperature,
        device=device,
    )

    saved = 0
    for idx, seq in enumerate(generated, start=1):
        out_path = out_dir / f"sample_{idx:02d}.mid"
        if save_token_sequence_as_midi(tokenizer, seq, out_path):
            saved += 1

    print(f"Saved {saved} MIDI files to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--genre-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())