# file: src/training/train_transformer.py

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.models.transformer import MusicTransformer
from src.training.utils import build_loader, load_npz_dataset, pick_device, save_curve


def train(args: argparse.Namespace) -> None:
    x, y = load_npz_dataset(args.data)

    device = pick_device(args.device)
    loader = build_loader(x, y, batch_size=args.batch_size)

    vocab_size = int(x.max().item()) + 1
    num_genres = int(y.max().item()) + 1
    model = MusicTransformer(vocab_size=vocab_size, num_genres=num_genres).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    print(f"Starting Transformer training on device={device}")
    print(
        f"Dataset size={x.size(0)} sequences, seq_len={x.size(1)}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, "
        f"vocab_size={vocab_size}, num_genres={num_genres}"
    )

    epoch_losses: list[float] = []
    epoch_perplexities: list[float] = []

    for epoch_idx in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        running = 0.0
        n = 0

        progress = tqdm(loader, desc=f"Epoch {epoch_idx}/{args.epochs}", leave=False, mininterval=1.0)
        for xb, gb in progress:
            xb = xb.to(device)
            gb = gb.to(device)

            inp = xb[:, :-1]
            tgt = xb[:, 1:]

            logits = model(inp, gb)
            loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1
            if n % 100 == 0:
             progress.set_postfix(loss=running / max(1, n))

        epoch_loss = running / max(1, n)
        epoch_perplexity = float(np.exp(epoch_loss))

        epoch_losses.append(epoch_loss)
        epoch_perplexities.append(epoch_perplexity)

        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch_idx}/{args.epochs} - "
            f"loss: {epoch_loss:.8f} - perplexity: {epoch_perplexity:.8f} - "
            f"time: {elapsed:.1f}s"
        )

    out_root = Path(args.out)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_root / "checkpoints" / "transformer.pt")
    save_curve(epoch_losses, out_root / "plots" / "transformer_loss.json")
    save_curve(epoch_perplexities, out_root / "plots" / "transformer_perplexity.json")

    print(f"Saved checkpoint to: {out_root / 'checkpoints' / 'transformer.pt'}")
    print(f"Saved loss curve to: {out_root / 'plots' / 'transformer_loss.json'}")
    print(f"Saved perplexity curve to: {out_root / 'plots' / 'transformer_perplexity.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(args)