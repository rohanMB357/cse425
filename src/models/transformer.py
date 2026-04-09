from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_genres: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.genre_emb = nn.Embedding(max(1, num_genres), d_model)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor, genre_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_emb(x)
        g = self.genre_emb(genre_ids).unsqueeze(1)
        h = self.pos(tok + g)
        mask = self._causal_mask(x.size(1), x.device)
        y = self.encoder(h, mask=mask)
        return self.out(y)

    @torch.no_grad()
    def generate(
        self,
        seed_tokens: torch.Tensor,
        genre_id: int,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        out = seed_tokens.clone()
        for _ in range(max_new_tokens):
            genre = torch.full((out.size(0),), genre_id, device=out.device, dtype=torch.long)
            logits = self(out, genre)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out
