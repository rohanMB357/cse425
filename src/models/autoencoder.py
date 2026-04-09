from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        _, (h_n, _) = self.encoder(emb)
        z = self.to_latent(h_n[-1])

        h0 = torch.tanh(self.from_latent(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        dec_out, _ = self.decoder(emb, (h0, c0))
        logits = self.out(dec_out)
        return logits, z
