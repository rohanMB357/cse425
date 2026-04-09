from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.generation.midi_export import tokens_to_midi_notes
from src.models.transformer import MusicTransformer
from src.preprocessing.tokenizer import MusicTokenizer


def main(args: argparse.Namespace) -> None:
    tk = MusicTokenizer.from_vocab(args.vocab)
    meta = np.load(args.data)
    x = torch.from_numpy(meta["x"]).long()
    y = torch.from_numpy(meta["y"]).long()

    vocab_size = int(x.max().item()) + 1
    num_genres = int(y.max().item()) + 1

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = MusicTransformer(vocab_size=vocab_size, num_genres=num_genres).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    idx = torch.randint(0, x.size(0), (args.n_samples,))
    seed = x[idx, : args.seed_len].to(device)

    generated = model.generate(seed, genre_id=args.genre_id, max_new_tokens=args.new_tokens, temperature=args.temperature)
    generated_np = generated.cpu().numpy()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, seq in enumerate(generated_np):
        notes = tk.decode_tokens(seq.tolist())
        tokens_to_midi_notes(notes, out_dir / f"sample_{i:02d}.mid")

    np.savez(out_dir / "generated_tokens.npz", x=generated_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--genre-id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
