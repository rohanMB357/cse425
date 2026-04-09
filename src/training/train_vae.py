from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.models.vae import LSTMVAE, kl_divergence
from src.training.utils import build_loader, load_npz_dataset, pick_device, save_curve


def train(args: argparse.Namespace) -> None:
    x, _ = load_npz_dataset(args.data)

    device = pick_device(args.device)
    loader = build_loader(x, torch.zeros(x.size(0), dtype=torch.long), batch_size=args.batch_size)

    model = LSTMVAE(vocab_size=int(x.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    for _epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, _ in loader:
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
            n += 1
        epoch_losses.append(running / max(1, n))

    out_root = Path(args.out)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_root / "checkpoints" / "vae.pt")
    save_curve(epoch_losses, out_root / "plots" / "vae_loss.json")


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
