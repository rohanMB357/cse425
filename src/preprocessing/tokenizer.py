import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class TokenizerConfig:
    pitch_min: int = 21
    pitch_max: int = 108
    velocity_bins: int = 8
    duration_bins: Tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


class MusicTokenizer:
    def __init__(self, cfg: TokenizerConfig | None = None):
        self.cfg = cfg or TokenizerConfig()
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>"]
        self.pitch_tokens = [f"PITCH_{p}" for p in range(self.cfg.pitch_min, self.cfg.pitch_max + 1)]
        self.velocity_tokens = [f"VEL_{i}" for i in range(self.cfg.velocity_bins)]
        self.duration_tokens = [f"DUR_{d}" for d in self.cfg.duration_bins]

        vocab = self.special_tokens + self.pitch_tokens + self.velocity_tokens + self.duration_tokens
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.itos: Dict[int, str] = {i: t for t, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def _velocity_to_bin(self, velocity: int) -> int:
        velocity = int(np.clip(velocity, 1, 127))
        return min(self.cfg.velocity_bins - 1, int((velocity / 128) * self.cfg.velocity_bins))

    def _duration_to_bin(self, duration_beats: float) -> float:
        arr = np.array(self.cfg.duration_bins)
        idx = int(np.argmin(np.abs(arr - duration_beats)))
        return float(arr[idx])

    def encode_events(self, notes: Sequence[Tuple[int, int, float]]) -> List[int]:
        tokens = [self.stoi["<BOS>"]]
        for pitch, velocity, duration_beats in notes:
            p = int(np.clip(pitch, self.cfg.pitch_min, self.cfg.pitch_max))
            v_bin = self._velocity_to_bin(velocity)
            d_bin = self._duration_to_bin(duration_beats)
            tokens.extend(
                [
                    self.stoi[f"PITCH_{p}"],
                    self.stoi[f"VEL_{v_bin}"],
                    self.stoi[f"DUR_{d_bin}"],
                ]
            )
        tokens.append(self.stoi["<EOS>"])
        return tokens

    def decode_tokens(self, token_ids: Sequence[int]) -> List[Tuple[int, int, float]]:
        notes: List[Tuple[int, int, float]] = []
        triplet: List[str] = []
        for token_id in token_ids:
            token = self.itos.get(int(token_id), "<PAD>")
            if token in self.special_tokens:
                continue
            triplet.append(token)
            if len(triplet) == 3:
                pitch = int(triplet[0].split("_")[1])
                vel_bin = int(triplet[1].split("_")[1])
                duration = float(triplet[2].split("_")[1])
                velocity = int((vel_bin + 0.5) / self.cfg.velocity_bins * 127)
                notes.append((pitch, velocity, duration))
                triplet = []
        return notes

    def save_vocab(self, path: str | Path) -> None:
        payload = {
            "stoi": self.stoi,
            "cfg": {
                "pitch_min": self.cfg.pitch_min,
                "pitch_max": self.cfg.pitch_max,
                "velocity_bins": self.cfg.velocity_bins,
                "duration_bins": list(self.cfg.duration_bins),
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def from_vocab(path: str | Path) -> "MusicTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = TokenizerConfig(
            pitch_min=payload["cfg"]["pitch_min"],
            pitch_max=payload["cfg"]["pitch_max"],
            velocity_bins=payload["cfg"]["velocity_bins"],
            duration_bins=tuple(payload["cfg"]["duration_bins"]),
        )
        tk = MusicTokenizer(cfg)
        tk.stoi = {k: int(v) for k, v in payload["stoi"].items()}
        tk.itos = {v: k for k, v in tk.stoi.items()}
        return tk
