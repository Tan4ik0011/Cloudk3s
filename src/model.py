import torch
import torch.nn as nn
from typing import Tuple, List


class ConditionalVAE(nn.Module):
    def __init__(self,
                 window_size: int = 32,
                 num_features: int = 3,
                 target_idx: int = 2,
                 latent_dim: int = 8,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64)) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.target_idx = target_idx
        self.cond_idx = tuple(idx for idx in range(num_features) if idx != target_idx)

        self.target_dim = window_size
        self.cond_dim = window_size * len(self.cond_idx)
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)

        encoder_input_dim = self.target_dim + self.cond_dim
        decoder_input_dim = self.latent_dim + self.cond_dim

        self.encoder = self._make_encoder(encoder_input_dim, self.hidden_dims)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.decoder = self._make_decoder(decoder_input_dim, self.hidden_dims, self.target_dim)

    @staticmethod
    def _make_encoder(input_dim: int, hidden_dims: Tuple[int, ...]) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        return nn.Sequential(*layers)

    @staticmethod
    def _make_decoder(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def flatten_batch(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1)

    def split_inputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target = x[..., self.target_idx]
        condition = x[..., self.cond_idx]
        return target, condition

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        dec_input = torch.cat([z, self.flatten_batch(condition)], dim=1)
        recon = self.decoder(dec_input)
        return recon.view(-1, self.window_size)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target, condition = self.split_inputs(x)
        enc_input = torch.cat([self.flatten_batch(target), self.flatten_batch(condition)], dim=1)
        hidden = self.encoder(enc_input)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        recon = self.decode(z, condition)
        return recon, target, mu, logvar