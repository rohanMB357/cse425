from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.autoencoder import LSTMAutoencoder
from src.training.utils import load_npz_dataset, pick_device, save_curve


def _build_split_loaders(
    x: torch.Tensor,
    batch_size: int,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    n = x.size(0)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    train_n = int(n * train_ratio)

    x_train = x[perm[:train_n]]
    x_valid = x[perm[train_n:]]

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x_valid), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def _run_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ce: nn.CrossEntropyLoss,
    device: torch.device,
    epoch_label: str,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)

    running = 0.0
    steps = 0
    progress = tqdm(loader, desc=epoch_label, leave=False)
    for (xb,) in progress:
        xb = xb.to(device)
        logits, _ = model(xb)
        loss = ce(logits.reshape(-1, logits.size(-1)), xb.reshape(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running += float(loss.item())
        steps += 1
        progress.set_postfix(loss=running / max(1, steps))

    return running / max(1, steps)


def train(args: argparse.Namespace) -> None:
    x, _ = load_npz_dataset(args.data)
    device = pick_device(args.device)

    if args.max_samples > 0:
        x = x[: args.max_samples]

    train_loader, valid_loader = _build_split_loaders(
        x,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    model = LSTMAutoencoder(
        vocab_size=int(x.max().item()) + 1,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    print(f"Starting AE training on device={device}")
    print(
        f"Dataset size={x.size(0)} sequences, seq_len={x.size(1)}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}"
    )

    train_losses: list[float] = []
    valid_losses: list[float] = []
    best_valid = float("inf")

    out_root = Path(args.out)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    for epoch_idx in range(1, args.epochs + 1):
        start = time.perf_counter()
        tr_loss = _run_epoch(model, train_loader, opt, ce, device, f"Train {epoch_idx}/{args.epochs}")
        va_loss = _run_epoch(model, valid_loader, None, ce, device, f"Valid {epoch_idx}/{args.epochs}")

        train_losses.append(tr_loss)
        valid_losses.append(va_loss)
        elapsed = time.perf_counter() - start
        print(
            f"Epoch {epoch_idx}/{args.epochs} - "
            f"train_loss: {tr_loss:.8f} - valid_loss: {va_loss:.8fdir outputs\checkpoints
dir outputs\plots} - time: {elapsed:.1f}s"
        )

        if va_loss < best_valid:
            best_valid = va_loss
            torch.save(model.state_dict(), out_root / "checkpoints" / "ae.pt")

    save_curve(train_losses, out_root / "plots" / "ae_train_loss.json")
    save_curve(valid_losses, out_root / "plots" / "ae_valid_loss.json")
    print(f"Saved checkpoint to: {out_root / 'checkpoints' / 'ae.pt'}")
    print(f"Saved train curve to: {out_root / 'plots' / 'ae_train_loss.json'}")
    print(f"Saved valid curve to: {out_root / 'plots' / 'ae_valid_loss.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    train(args)