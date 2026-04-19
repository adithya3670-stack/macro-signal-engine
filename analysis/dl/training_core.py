from __future__ import annotations

import os
from typing import Any, Callable, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from backend.shared.device import use_amp, use_pin_memory


def train_model_instance(
    *,
    device: torch.device,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_type: str,
    params: Dict[str, Any],
    top_features,
    save_path: str,
    force_full_training: bool,
    window_size: int,
    manual_epochs=None,
    min_epochs: int = 15,
    lstm_cls: Callable[..., torch.nn.Module],
    transformer_cls: Callable[..., torch.nn.Module],
    nbeats_cls: Callable[..., torch.nn.Module],
    dataset_cls: Callable[..., Any],
    loss_cls: Callable[..., torch.nn.Module],
):
    """
    Train a single DL model instance/fold and return evaluation metrics.
    """
    input_size = len(top_features)

    try:
        unique_classes = np.unique(y_train)
        if len(unique_classes) > 1:
            class_weights = compute_class_weight(
                "balanced",
                classes=np.array([0, 1]),
                y=y_train.flatten(),
            )
            cw_val = class_weights[1] / class_weights[0]
        else:
            cw_val = 1.0
        _pos_weight = torch.tensor([cw_val], device=device)
    except Exception:
        _pos_weight = torch.tensor([1.0], device=device)

    train_dataset = dataset_cls(X_train, y_train)
    val_dataset = dataset_cls(X_val, y_val)
    test_dataset = dataset_cls(X_test, y_test)

    workers = 0
    batch_size = params.get("batch_size", 512)
    pin_memory = use_pin_memory(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=workers,
    )

    if model_type == "transformer":
        model = transformer_cls(
            input_size=input_size,
            d_model=params["trans_d_model"],
            nhead=params["trans_nhead"],
            num_layers=params["trans_layers"],
            dropout=params.get("trans_dropout", params.get("dropout", 0.1)),
        ).to(device)
    elif model_type == "nbeats":
        model = nbeats_cls(
            num_features=input_size,
            window_size=window_size,
            num_stacks=params["nb_stacks"],
            num_blocks=params["nb_blocks"],
            layer_width=params["nb_width"],
        ).to(device)
    else:
        model = lstm_cls(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            num_heads=16,
            dropout=params["dropout"],
        ).to(device)

    gamma_val = params.get("focal_gamma", 2.0)
    criterion = loss_cls(alpha=0.25, gamma=gamma_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler_amp = torch.amp.GradScaler("cuda") if use_amp(device) else None

    best_loss = float("inf")
    patience = 10 if not force_full_training else 999
    patience_counter = 0
    optimal_epochs_found = min_epochs
    accumulation_steps = 2

    max_epochs = int(params.get("epochs_override", 50))
    if force_full_training and manual_epochs:
        max_epochs = int(manual_epochs)

    for _epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if torch.isnan(X_batch).any():
                continue

            if scaler_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss = loss / accumulation_steps
                scaler_amp.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    optimizer.zero_grad()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss = loss / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if not force_full_training:
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                optimal_epochs_found = _epoch + 1
                torch.save(model.state_dict(), save_path)
            else:
                if _epoch >= min_epochs:
                    patience_counter += 1

            if patience_counter >= patience and _epoch >= min_epochs:
                break
        else:
            torch.save(model.state_dict(), save_path)
            optimal_epochs_found = max_epochs

    if not force_full_training and os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path))
        except Exception:
            pass

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_probs = 1 / (1 + np.exp(-np.array(all_preds)))
    all_labels = np.array(all_labels)

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.3, 0.75, 0.05):
        binary = (all_probs > thr).astype(int)
        f1 = f1_score(all_labels, binary, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    final_binary = (all_probs > best_thr).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, final_binary)),
        "precision": float(precision_score(all_labels, final_binary, zero_division=0)),
        "recall": float(recall_score(all_labels, final_binary, zero_division=0)),
        "f1": float(best_f1),
        "auc": float(roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5),
        "threshold": best_thr,
        "optimal_epochs": max(optimal_epochs_found, 15) if not force_full_training else optimal_epochs_found,
    }
    return metrics
