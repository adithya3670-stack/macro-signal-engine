import math
from typing import Dict

import torch
import torch.nn as nn

from analysis.deep_learning_model import LSTMAttentionModel, NBeatsNet


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        x_norm = (x - mean) / std
        return (x_norm * self.gamma) + self.beta


class NLinearRegressor(nn.Module):
    def __init__(self, input_size: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.input_size = input_size
        self.linear = nn.Linear(window_size * input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x[:, -1:, :]
        return self.linear(centered.reshape(centered.size(0), -1))


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        window_size: int,
        patch_len: int = 5,
        stride: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_len = min(patch_len, window_size)
        self.stride = max(1, stride)
        token_dim = input_size * self.patch_len

        self.patch_proj = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.patch_len:
            pad = self.patch_len - x.size(1)
            x = torch.nn.functional.pad(x, (0, 0, pad, 0))

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().permute(0, 1, 3, 2).reshape(x.size(0), -1, self.patch_len * x.size(2))
        tokens = self.patch_proj(patches)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class TiDEResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TiDERegressor(nn.Module):
    def __init__(self, input_size: int, window_size: int, hidden_size: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        flat_dim = input_size * window_size
        self.input_proj = nn.Linear(flat_dim, hidden_size)
        self.blocks = nn.Sequential(*[TiDEResidualBlock(hidden_size, dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        hidden = self.input_proj(flat)
        hidden = self.blocks(hidden)
        return self.head(hidden)


class NHiTSRegressor(nn.Module):
    def __init__(self, input_size: int, window_size: int, hidden_size: int = 128, pool_sizes=None, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.input_size = input_size
        self.pool_sizes = list(pool_sizes or [1, 2, 5])
        self.projections = nn.ModuleList()
        for pool in self.pool_sizes:
            pooled_steps = math.ceil(window_size / pool)
            self.projections.append(
                nn.Sequential(
                    nn.Linear(pooled_steps * input_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cf = x.transpose(1, 2)
        mixed = None
        for pool, proj in zip(self.pool_sizes, self.projections):
            pooled = torch.nn.functional.avg_pool1d(x_cf, kernel_size=pool, stride=pool, ceil_mode=True)
            hidden = proj(pooled.reshape(x.size(0), -1))
            mixed = hidden if mixed is None else mixed + hidden
        return self.head(mixed)


class RevINLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.revin = RevIN(input_size)
        self.model = LSTMAttentionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=4,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.revin(x))


class NBeatsRegressor(nn.Module):
    def __init__(self, input_size: int, window_size: int, num_stacks: int = 2, num_blocks: int = 3, layer_width: int = 128, dropout: float = 0.1):
        super().__init__()
        self.model = NBeatsNet(
            num_features=input_size,
            window_size=window_size,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            layer_width=layer_width,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_price_model(model_type: str, input_size: int, config: Dict[str, float]) -> nn.Module:
    window_size = int(config["window_size"])

    if model_type == "nlinear":
        return NLinearRegressor(input_size=input_size, window_size=window_size)
    if model_type == "patchtst":
        return PatchTSTRegressor(
            input_size=input_size,
            window_size=window_size,
            patch_len=int(config.get("patch_len", 5)),
            stride=int(config.get("stride", 2)),
            d_model=int(config.get("d_model", 64)),
            nhead=int(config.get("nhead", 4)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_type == "tide":
        return TiDERegressor(
            input_size=input_size,
            window_size=window_size,
            hidden_size=int(config.get("hidden_size", 128)),
            depth=int(config.get("depth", 3)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_type == "nhits":
        return NHiTSRegressor(
            input_size=input_size,
            window_size=window_size,
            hidden_size=int(config.get("hidden_size", 128)),
            pool_sizes=config.get("pool_sizes", [1, 2, 5]),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_type == "lstm_reg_revin":
        return RevINLSTMRegressor(
            input_size=input_size,
            hidden_size=int(config.get("hidden_size", 128)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.2)),
        )
    if model_type == "nbeats_reg":
        return NBeatsRegressor(
            input_size=input_size,
            window_size=window_size,
            num_stacks=int(config.get("nb_stacks", 2)),
            num_blocks=int(config.get("nb_blocks", 3)),
            layer_width=int(config.get("nb_width", 128)),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported price-3d model type: {model_type}")
