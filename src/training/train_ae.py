from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.models.autoencoder import LSTMAutoencoder
from src.training.utils import build_loader, load_npz_dataset, pick_device, save_curve


def train(args: argparse.Namespace) -> None:
    x, y = load_npz_dataset(args.data)
    del y

    device = pick_device(args.device)
    loader = build_loader(x, torch.zeros(x.size(0), dtype=torch.long), batch_size=args.batch_size)

    model = LSTMAutoencoder(vocab_size=int(x.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    epoch_losses: list[float] = []
    for _epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, _ in loader:
            xb = xb.to(device)
            inp = xb[:, :-1]
            tgt = xb[:, 1:]

            logits, _z = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1
        epoch_losses.append(running / max(1, n))

    out_root = Path(args.out)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_root / "checkpoints" / "autoencoder.pt")
    save_curve(epoch_losses, out_root / "plots" / "ae_loss.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    Path(args.out, "plots").mkdir(parents=True, exist_ok=True)
    train(args)
