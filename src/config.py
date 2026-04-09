from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    seq_len: int = 128
    step: int = 16
    min_notes_per_file: int = 64


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
