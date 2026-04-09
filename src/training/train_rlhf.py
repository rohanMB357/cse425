from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.models.transformer import MusicTransformer
from src.training.utils import load_npz_dataset, pick_device


def simulated_reward(tokens: torch.Tensor) -> torch.Tensor:
    unique_ratio = torch.tensor([len(torch.unique(t)) / max(1, t.numel()) for t in tokens], device=tokens.device)
    return unique_ratio


def load_human_rewards(csv_path: str | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
    if not csv_path:
        return None
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "score" not in df.columns:
        return None
    vals = torch.tensor(df["score"].values[:batch_size], dtype=torch.float32, device=device)
    vals = (vals - 1.0) / 4.0
    return vals


def train(args: argparse.Namespace) -> None:
    x, y = load_npz_dataset(args.data)
    device = pick_device(args.device)

    vocab_size = int(x.max().item()) + 1
    num_genres = int(y.max().item()) + 1

    model = MusicTransformer(vocab_size=vocab_size, num_genres=num_genres).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for _ in range(args.steps):
        idx = torch.randint(0, x.size(0), (args.batch_size,))
        seed = x[idx, : args.seed_len].to(device)
        genre = y[idx].to(device)

        generated = model.generate(seed, genre_id=int(genre[0].item()), max_new_tokens=args.new_tokens)
        logits = model(generated[:, :-1], genre)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        sampled = generated[:, -1]
        chosen_lp = log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)

        rewards = load_human_rewards(args.human_csv, generated.size(0), device)
        if rewards is None or rewards.numel() < generated.size(0):
            rewards = simulated_reward(generated)

        loss = -(rewards.detach() * chosen_lp).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    out_root = Path(args.out)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_root / "checkpoints" / "transformer_rlhf.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--human-csv", type=str, default="")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(args)
