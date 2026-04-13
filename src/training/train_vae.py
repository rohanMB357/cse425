from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.models.vae import LSTMVAE, kl_divergence
from src.training.utils import build_loader, load_npz_dataset, pick_device, save_curve


def train(args: argparse.Namespace) -> None:
    x, _ = load_npz_dataset(args.data)

    device = pick_device(args.device)
    loader = build_loader(x, torch.zeros(x.size(0), dtype=torch.long), batch_size=args.batch_size)

    model = LSTMVAE(vocab_size=int(x.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    print(f"Starting VAE training on device={device}")
    print(
        f"Dataset size={x.size(0)} sequences, seq_len={x.size(1)}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, beta={args.beta}"
    )

    epoch_losses: list[float] = []
    for epoch_idx in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        running = 0.0
        running_recon = 0.0
        running_kl = 0.0
        n = 0
        progress = tqdm(loader, desc=f"Epoch {epoch_idx}/{args.epochs}", leave=False)
        for xb, _ in progress:
            xb = xb.to(device)
            inp = xb[:, :-1]
            tgt = xb[:, 1:]

            logits, mu, logvar, _ = model(inp)
            recon = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            kl = kl_divergence(mu, logvar)
            loss = recon + args.beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            running_recon += float(recon.item())
            running_kl += float(kl.item())
            n += 1
            progress.set_postfix(loss=running / n)
        epoch_loss = running / max(1, n)
        epoch_recon = running_recon / max(1, n)
        epoch_kl = running_kl / max(1, n)
        epoch_losses.append(epoch_loss)
        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch_idx}/{args.epochs} - "
            f"loss: {epoch_loss:.4f} - recon: {epoch_recon:.4f} - kl: {epoch_kl:.4f} - "
            f"time: {elapsed:.1f}s"
        )

    out_root = Path(args.out)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_root / "checkpoints" / "vae.pt")
    save_curve(epoch_losses, out_root / "plots" / "vae_loss.json")
    print(f"Saved checkpoint to: {out_root / 'checkpoints' / 'vae.pt'}")
    print(f"Saved loss curve to: {out_root / 'plots' / 'vae_loss.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    Path(args.out, "plots").mkdir(parents=True, exist_ok=True)
    train(args)
