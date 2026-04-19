from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LSTMAttentionModel(nn.Module):
    """
    LSTM + Multi-Head Attention - optimized for high GPU utilization.
    """

    def __init__(self, input_size, hidden_size=256, num_layers=2, num_heads=16, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln1(lstm_out)
        lstm_out = self.dropout1(lstm_out)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out + lstm_out
        attn_out = self.ln2(attn_out)
        attn_out = self.dropout2(attn_out)

        pooled = attn_out.mean(dim=1)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerTimeSeriesModel(nn.Module):
    """
    Encoder-only transformer for time-series classification.
    """

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.fc_out(x)
        return out


class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance.
    """

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, theta_dim, num_layers, layer_width, dropout_p=0.2):
        super().__init__()

        layers = [
            nn.Linear(input_dim, layer_width),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(layer_width, layer_width),
                    nn.ReLU(),
                    nn.Dropout(dropout_p),
                ],
            )
        self.fc_stack = nn.Sequential(*layers)

        self.backcast_head = nn.Linear(layer_width, input_dim)
        self.theta_head = nn.Linear(layer_width, theta_dim)

    def forward(self, x):
        h = self.fc_stack(x)
        backcast = self.backcast_head(h)
        theta = self.theta_head(h)
        return backcast, theta


class NBeatsNet(nn.Module):
    def __init__(self, num_features, window_size, num_stacks=2, num_blocks=3, layer_width=128, dropout=0.2):
        super().__init__()

        self.input_dim = num_features * window_size
        self.theta_dim = 64
        self.stacks = nn.ModuleList()

        for _ in range(num_stacks):
            blocks = nn.ModuleList()
            for _ in range(num_blocks):
                blocks.append(NBeatsBlock(self.input_dim, self.theta_dim, 3, layer_width, dropout))
            self.stacks.append(blocks)

        self.final_fc = nn.Linear(self.theta_dim, 1)

    def forward(self, x):
        batch_size, _window, _features = x.size()
        x_flat = x.view(batch_size, -1)

        residual = x_flat
        forecast = torch.zeros(batch_size, self.theta_dim).to(x.device)
        for stack in self.stacks:
            for block in stack:
                backcast, theta = block(residual)
                residual = residual - backcast
                forecast = forecast + theta

        return self.final_fc(forecast)


__all__ = [
    "FocalLoss",
    "LSTMAttentionModel",
    "NBeatsBlock",
    "NBeatsNet",
    "PositionalEncoding",
    "TimeSeriesDataset",
    "TransformerTimeSeriesModel",
]
