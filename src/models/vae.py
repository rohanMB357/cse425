from __future__ import annotations

import torch
from torch import nn


class LSTMVAE(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        _, (h_n, _) = self.encoder(emb)
        h = h_n[-1]
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        h0 = torch.tanh(self.from_latent(z)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(emb, (h0, c0))
        return self.out(dec_out)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(x, z)
        return logits, mu, logvar, z


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
