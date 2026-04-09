from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_npz_dataset(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path)
    x = torch.from_numpy(data["x"]).long()
    y = torch.from_numpy(data["y"]).long()
    return x, y


def build_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def pick_device(device_arg: str = "cuda") -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_curve(curve: list[float], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps({"values": curve}, indent=2), encoding="utf-8")
