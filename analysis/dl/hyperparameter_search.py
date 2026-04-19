from __future__ import annotations

import gc
import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from analysis.dl.model_definitions import (
    FocalLoss,
    LSTMAttentionModel,
    NBeatsNet,
    TimeSeriesDataset,
    TransformerTimeSeriesModel,
)


def _safe_train_step(model, optimizer, criterion, X_batch, y_batch, device):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    optimizer.zero_grad()
    if torch.cuda.is_available():
        with torch.amp.autocast("cuda"):
            out = model(X_batch)
            loss = criterion(out, y_batch)
    else:
        out = model(X_batch)
        loss = criterion(out, y_batch)
    loss.backward()
    optimizer.step()


def optimize_models_core(
    *,
    dl_model,
    model_type: str = "lstm",
    epochs: int = 100,  # kept for backward compatibility
    batch_size: int = 32,  # kept for backward compatibility
    progress_callback=None,
    save_config: bool = True,
    base_config: Optional[Dict[str, Any]] = None,
    train_cutoff_date: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Random-search optimization core extracted from DLMacroModel monolith.
    """
    _ = epochs, batch_size
    iterations = kwargs.get("iterations", 10)

    print(f"--- Starting Optimization for {model_type.upper()} ({iterations} iters) ---")
    df = dl_model.load_and_preprocess()

    if train_cutoff_date:
        print(f"  [Optimization] Applying Cutoff: {train_cutoff_date}")
        df = df.loc[:train_cutoff_date]

    param_space = {
        "window_size": [30, 45, 60, 90],
        "hidden_size": [64, 128, 256],
        "num_layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.001, 0.0005],
        "num_features": [10, 20, 30, 50],
        "batch_size": [128, 256, 512],
        "focal_gamma": [1.0, 2.0, 3.0],
    }

    if model_type == "transformer":
        param_space.update(
            {
                "learning_rate": [0.001, 0.0005, 0.0001],
                "trans_d_model": [64, 128, 256, 512],
                "trans_nhead": [4, 8, 16],
                "trans_layers": [2, 3, 4, 6],
            },
        )

    if model_type == "nbeats":
        param_space.update(
            {
                "nb_stacks": [2, 3, 4],
                "nb_blocks": [1, 3, 5],
                "nb_width": [128, 256, 512, 1024],
            },
        )

    exclude_cols = [c for c in df.columns if "Target_" in c or "Regime_" in c or "Date" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    scaler = MinMaxScaler()
    train_split_idx = int(len(df) * 0.70)
    scaler.fit(df[feature_cols].iloc[:train_split_idx])
    X_all_scaled_df = pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols)

    best_configs: Dict[str, Dict[str, Any]] = {}
    if base_config:
        best_configs = base_config.copy()
    elif os.path.exists(f"{dl_model.model_dir}/dl_config.json"):
        try:
            with open(f"{dl_model.model_dir}/dl_config.json", "r", encoding="utf-8") as handle:
                best_configs = json.load(handle)
        except Exception:
            pass

    total_steps = len(dl_model.assets) * len(dl_model.horizons) * iterations
    current_step = 0

    for asset in dl_model.assets:
        for horizon in dl_model.horizons:
            target_col = f"Target_{asset}_{horizon}"
            if target_col not in df.columns:
                continue

            train_df = df.iloc[:train_split_idx]
            correlations = train_df[feature_cols].corrwith(train_df[target_col]).abs().sort_values(ascending=False)
            y_raw = (df[target_col] > 0).astype(int).values
            valid_mask = df[target_col].notna().values

            best_f1 = -1.0
            best_params = None
            print(f"Optimizing {asset} {horizon}...")

            for i in range(iterations):
                model = None
                optimizer = None
                criterion = None
                train_loader = None
                val_loader = None

                current_step += 1

                p_window = random.choice(param_space["window_size"])
                p_hidden = random.choice(param_space["hidden_size"])
                p_layers = random.choice(param_space["num_layers"])
                p_drop = random.choice(param_space["dropout"])
                p_lr = random.choice(param_space["learning_rate"])
                p_num_feat = random.choice(param_space["num_features"])
                p_batch = random.choice(param_space["batch_size"])
                p_gamma = random.choice(param_space["focal_gamma"])

                p_d_model = random.choice(param_space.get("trans_d_model", [128]))
                p_nhead = random.choice(param_space.get("trans_nhead", [8]))
                if p_d_model % p_nhead != 0:
                    p_nhead = 4
                p_trans_layers = random.choice(param_space.get("trans_layers", [3]))

                msg = f"Testing {asset} {horizon} Iter {i+1}: W={p_window} Feats={p_num_feat} Batch={p_batch}"
                if progress_callback:
                    progress_callback(int(current_step / total_steps * 100), msg)

                top_features = correlations.head(p_num_feat).index.tolist()
                X_selected_values = X_all_scaled_df[top_features].values

                X_seq, y_seq = dl_model.create_sequences(X_selected_values, p_window, y_raw)
                valid_indices = [idx for idx in range(len(y_seq)) if valid_mask[idx + p_window]]
                if len(valid_indices) < 100:
                    continue

                X_final = X_seq[valid_indices]
                y_final = y_seq[valid_indices]

                t_idx = int(len(X_final) * 0.70)
                v_idx = int(len(X_final) * 0.85)

                X_train = X_final[:t_idx]
                y_train = y_final[:t_idx]
                X_val = X_final[t_idx:v_idx]
                y_val = y_final[t_idx:v_idx]

                workers = 0
                train_loader = DataLoader(
                    TimeSeriesDataset(X_train, y_train),
                    batch_size=p_batch,
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                    persistent_workers=(workers > 0),
                )
                val_loader = DataLoader(
                    TimeSeriesDataset(X_val, y_val),
                    batch_size=p_batch,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    persistent_workers=(workers > 0),
                )

                if model_type == "transformer":
                    model = TransformerTimeSeriesModel(len(top_features), p_d_model, p_nhead, p_trans_layers, p_drop).to(dl_model.device)
                elif model_type == "nbeats":
                    p_nb_stacks = random.choice(param_space.get("nb_stacks", [2]))
                    p_nb_blocks = random.choice(param_space.get("nb_blocks", [3]))
                    p_nb_width = random.choice(param_space.get("nb_width", [128]))
                    model = NBeatsNet(len(top_features), p_window, p_nb_stacks, p_nb_blocks, p_nb_width, p_drop).to(dl_model.device)
                else:
                    model = LSTMAttentionModel(len(top_features), p_hidden, p_layers, 16, p_drop).to(dl_model.device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=p_lr)
                criterion = FocalLoss(alpha=0.25, gamma=p_gamma)

                model.train()
                for _epoch in range(5):
                    for X_b, y_b in train_loader:
                        _safe_train_step(model, optimizer, criterion, X_b, y_b, dl_model.device)

                model.eval()
                preds, labels = [], []
                with torch.no_grad():
                    for X_b, y_b in val_loader:
                        out = model(X_b.to(dl_model.device))
                        preds.extend(out.cpu().numpy().flatten())
                        labels.extend(y_b.cpu().numpy().flatten())

                best_iter_f1 = 0
                for thr in [0.4, 0.5, 0.6]:
                    probs = 1 / (1 + np.exp(-np.array(preds)))
                    binary = (probs > thr).astype(int)
                    score = f1_score(labels, binary, zero_division=0)
                    if score > best_iter_f1:
                        best_iter_f1 = score

                if best_iter_f1 > best_f1:
                    best_f1 = best_iter_f1
                    best_params = {
                        "window_size": p_window,
                        "hidden_size": p_hidden,
                        "num_layers": p_layers,
                        "dropout": p_drop,
                        "trans_d_model": p_d_model,
                        "trans_nhead": p_nhead,
                        "trans_layers": p_trans_layers,
                        "num_features": p_num_feat,
                        "batch_size": p_batch,
                        "focal_gamma": p_gamma,
                    }
                    if model_type == "nbeats":
                        best_params.update(
                            {
                                "nb_stacks": p_nb_stacks,
                                "nb_blocks": p_nb_blocks,
                                "nb_width": p_nb_width,
                            },
                        )
                    print(f"  New Best: F1={best_f1:.3f} (W={p_window}, Feats={p_num_feat})")

                del model, optimizer, criterion
                if train_loader:
                    del train_loader
                if val_loader:
                    del val_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if best_params:
                config_key = f"{asset}_{horizon}_{model_type}"
                best_configs[config_key] = best_params

    if save_config:
        with open(f"{dl_model.model_dir}/dl_config.json", "w", encoding="utf-8") as handle:
            json.dump(best_configs, handle, indent=4)

    return best_configs


class DeepLearningHyperparameterSearch:
    """Service boundary for DL optimization/search workflows."""

    def __init__(self, model_dir: Optional[str] = None, dl_model=None) -> None:
        self.model_dir = model_dir
        self.dl_model = dl_model

    def _get_model(self):
        if self.dl_model is not None:
            return self.dl_model
        from analysis.deep_learning_model import DLMacroModel

        return DLMacroModel(model_dir=self.model_dir) if self.model_dir else DLMacroModel()

    def run(
        self,
        model_type: str,
        iterations: int = 10,
        **kwargs: Any,
    ) -> Any:
        model = self._get_model()
        return optimize_models_core(
            dl_model=model,
            model_type=model_type,
            iterations=iterations,
            **kwargs,
        )
