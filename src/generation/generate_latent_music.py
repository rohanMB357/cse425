# file: src/generation/generate_latent_music.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.generation.midi_export import tokens_to_midi_notes
from src.models.autoencoder import LSTMAutoencoder
from src.models.vae import LSTMVAE
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


@torch.no_grad()
def generate_from_autoencoder(
    model: LSTMAutoencoder,
    seeds: torch.Tensor,
) -> list[list[int]]:
    model.eval()
    logits, _ = model(seeds)
    pred = torch.argmax(logits, dim=-1)
    return [seq.tolist() for seq in pred.cpu()]


@torch.no_grad()
def generate_from_vae(
    model: LSTMVAE,
    tokenizer: MusicTokenizer,
    n_samples: int,
    new_tokens: int,
    device: torch.device,
) -> list[list[int]]:
    model.eval()

    bos_id = tokenizer.stoi["<BOS>"]
    latent_dim = model.mu_head.out_features
    generated_sequences: list[list[int]] = []

    for _ in range(n_samples):
        z = torch.randn(1, latent_dim, device=device)
        seq = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        for _step in range(new_tokens):
            logits = model.decode(seq, z)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            seq = torch.cat([seq, next_token], dim=1)

        generated_sequences.append(seq.squeeze(0).tolist())

    return generated_sequences


def build_autoencoder(
    vocab_size: int,
    checkpoint: str,
    device: torch.device,
) -> LSTMAutoencoder:
    model = LSTMAutoencoder(vocab_size=vocab_size).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model


def build_vae(
    vocab_size: int,
    checkpoint: str,
    device: torch.device,
) -> LSTMVAE:
    model = LSTMVAE(vocab_size=vocab_size).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ae", "vae"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    tokenizer = MusicTokenizer.from_vocab(args.vocab)
    vocab_size = tokenizer.vocab_size
    x, _ = load_dataset(args.data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "ae":
        model = build_autoencoder(vocab_size, args.checkpoint, device)
        n_samples = min(args.n_samples, x.size(0))
        seeds = x[:n_samples].to(device)
        generated = generate_from_autoencoder(model, seeds)
    else:
        model = build_vae(vocab_size, args.checkpoint, device)
        generated = generate_from_vae(
            model=model,
            tokenizer=tokenizer,
            n_samples=args.n_samples,
            new_tokens=args.new_tokens,
            device=device,
        )

    saved = 0
    for idx, token_ids in enumerate(generated, start=1):
        out_path = out_dir / f"{args.model}_sample_{idx:02d}.mid"
        if save_token_sequence_as_midi(tokenizer, token_ids, out_path, tempo=args.tempo):
            saved += 1

    print(f"Saved {saved} MIDI files to: {out_dir}")


if __name__ == "__main__":
    main()