from __future__ import annotations

import torch
from torch import nn


class CausalTemporalEncoder(nn.Module):
    """Encode a history sequence without allowing attention to future positions."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values)
        encoded = encoded + sinusoidal_position_encoding(
            encoded.shape[1],
            encoded.shape[2],
            encoded.device,
            encoded.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(
                encoded.shape[1],
                encoded.shape[1],
                device=encoded.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return self.encoder(encoded, mask=causal_mask)[:, -1]


def sinusoidal_position_encoding(
    length: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / max(dimension, 1))
    )
    encoding = torch.zeros((1, length, dimension), device=device, dtype=torch.float32)
    encoding[0, :, 0::2] = torch.sin(position * frequencies)
    encoding[0, :, 1::2] = torch.cos(position * frequencies[: dimension // 2])
    return encoding.to(dtype=dtype)
