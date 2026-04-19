from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def train_all_models_core(
    *,
    dl_model,
    model_type: str = "lstm",
    progress_callback=None,
    train_cutoff_date: Optional[str] = None,
    epochs: int = 50,
    config_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    force_full_training: bool = False,
    manual_epochs_dict: Optional[Dict[str, int]] = None,
    use_bagging_ensemble: bool = False,
    n_folds: int = 3,
    output_folder: Optional[str] = None,
    **kwargs,
):
    """
    Orchestrate end-to-end DL training across assets/horizons.
    Extracted from DLMacroModel to keep route-adapter class thin.
    """
    # Print GPU info once at start
    print(f"\n{'=' * 60}")
    print(f"Training [{model_type.upper()}] on: {dl_model.device}")
    if dl_model.device.type == "cuda":
        device_index = dl_model.device.index if dl_model.device.index is not None else torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print("Mixed Precision: Enabled (AMP)")
    elif dl_model.device.type == "mps":
        print("Apple Silicon MPS backend active")
        print("Mixed Precision: Disabled (AMP is CUDA-only)")
    else:
        print("CPU mode active")
    print(f"{'=' * 60}\n")

    df = dl_model.load_and_preprocess()

    # Apply Hold-out Cutoff if provided
    if output_folder:
        # Direct override for custom save location (e.g., Sequential Training)
        save_dir = output_folder
        os.makedirs(save_dir, exist_ok=True)
        print(f"  [Custom] Using output folder: {save_dir}")
    elif train_cutoff_date:
        print(f"Applying Training Cutoff (DL): {train_cutoff_date}")
        df = df.loc[:train_cutoff_date]

        # --- HOLDOUT STRUCTURE IMPLEMENTATION ---
        try:
            cutoff_dt = pd.to_datetime(train_cutoff_date)
            year_str = str(cutoff_dt.year)

            # Base models directory logic
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            holdout_base = os.path.join(base_dir, "models", "holdout_dl")

            # Create Year Folder: models/holdout_dl/2022
            save_dir = os.path.join(holdout_base, year_str)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            print(f"  [Auto-Pilot] Switching Context -> {save_dir}")

        except Exception as exc:
            print(f"  [Error] Could not create holdout directory: {exc}")
            save_dir = dl_model.model_dir
    else:
        save_dir = dl_model.model_dir

    # Identify Features
    exclude_cols = [c for c in df.columns if "Target_" in c or "Regime_" in c or "Date" in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    print(f"Training on {len(feature_cols)} Features")
    results = []

    scaler = MinMaxScaler()

    # PREVENT DATA LEAKAGE:
    # Fit scaler ONLY on the training portion of the available data
    if force_full_training:
        # For Production: Fit on EVERYTHING
        train_split_idx = len(df)
        print("  [Master DL] Fitting Scaler on 100% Data")
    else:
        # For Discovery/Backtest: Standard 70% fit
        train_split_idx = int(len(df) * 0.70)

    scaler.fit(df[feature_cols].iloc[:train_split_idx])

    # Transform entire (cutoff) dataset for creating sequences
    X_all_scaled = scaler.transform(df[feature_cols])  # Scaled values

    # Save Scaler & Features to the Dynamic Directory
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    joblib.dump(feature_cols, f"{save_dir}/features.pkl")

    # SNAPSHOT: Save Train/Test Data for Reproducibility
    # We save the *Raw* dataframe segments (before scaling) to allow full inspection
    snapshot_train = df.iloc[:train_split_idx]
    snapshot_test = df.iloc[train_split_idx:]

    # Check if snapshot already exists to avoid unnecessary IO
    if not os.path.exists(f"{save_dir}/train_data.csv"):
        snapshot_train.to_csv(f"{save_dir}/train_data.csv")
        snapshot_test.to_csv(f"{save_dir}/test_data.csv")
        print(f"  [Snapshot] Saved data to {save_dir}")

    total_models = len(dl_model.assets) * len(dl_model.horizons)
    current_model = 0

    # Filenames based on dynamic dir
    metrics_file = f"{save_dir}/dl_metrics_{model_type}.json"

    # Dictionary to store selected features for each model
    model_features_map = {}

    for asset in dl_model.assets:
        for horizon in dl_model.horizons:
            current_model += 1

            # Get Config for this Horizon
            if config_dict:
                # Use passed config (Memory)
                key = f"{asset}_{horizon}_{model_type}"
                params = config_dict.get(key, {})
                # Apply defaults if missing
                if not params:
                    params = dl_model.get_horizon_config(horizon, model_type, asset=asset)[1]  # Fallback
                window_size = int(params.get("window_size", 60))
            else:
                # Load from File (Standard)
                window_size, params = dl_model.get_horizon_config(horizon, model_type, asset=asset)

            if progress_callback:
                pct = int((current_model / total_models) * 100)
                progress_callback(pct, f"Training {model_type.upper()} {asset} {horizon} (W={window_size})...")

            target_col = f"Target_{asset}_{horizon}"
            if target_col not in df.columns:
                continue

            # --- FEATURE SELECTION ---
            # Calculate correlation of all features with THIS target
            # taking only training portion to avoid leakage
            train_df = df.iloc[:train_split_idx]
            correlations = train_df[feature_cols].corrwith(train_df[target_col]).abs()

            # Select Top Features (Dynamic)
            num_feats = params.get("num_features", 20)
            top_features = correlations.sort_values(ascending=False).head(num_feats).index.tolist()

            # Store for Inference
            model_key = f"{asset}_{horizon}_{model_type}"
            model_features_map[model_key] = top_features

            print(
                f"Training {model_type.upper()}: {asset} - {horizon} "
                f"(Window: {window_size}, Layers: {params['num_layers']})",
            )
            print(f"  Selected Top 5 Feats: {top_features[:5]}")

            y_raw = (df[target_col] > 0).astype(int).values
            valid_mask = df[target_col].notna().values

            # Prepare X using ONLY selected features
            # We need to transform FIRST using the scaler fit on ALL features, then select columns
            # df_scaled was not created earlier, so let's transform on fly just for these cols?
            # No, scaler expects all cols.
            # So we transform ALL cols first for this batch, then slice.
            X_all_scaled = scaler.transform(df[feature_cols])
            X_df_scaled = pd.DataFrame(X_all_scaled, columns=feature_cols)
            X_selected_values = X_df_scaled[top_features].values

            X_seq, y_seq = dl_model.create_sequences(X_selected_values, window_size, y_raw)

            valid_indices = [i for i in range(len(y_seq)) if valid_mask[i + window_size]]

            X_final = X_seq[valid_indices]
            y_final = y_seq[valid_indices]

            if len(X_final) < 80:
                print(f"Not enough data: {len(X_final)}")
                continue

            # --- ENSEMBLE / TRAINING LOOP ---
            # Logic: If n_folds=1 (Speed Mode), we run ONLY Fold 1 (Mid-Gap).
            # Fold 1 uses [0-60%] AND [80-100%] for training. Validates on [60-80%].
            # This ensures the model sees the LATEST data while still validating on unseen mid-data.

            if use_bagging_ensemble:
                folds_to_run = [1] if n_folds == 1 else range(3)
                total_folds_concept = 3  # Always conceptually 3 folds
            else:
                folds_to_run = [0]
                total_folds_concept = 1

            ensemble_metrics = []

            for fold_idx in folds_to_run:
                current_save_path = f"{save_dir}/{asset}_{horizon}_{model_type}"

                if use_bagging_ensemble:
                    current_save_path += f"_fold{fold_idx}.pth"
                    print(f"  [Ensemble] Training Fold {fold_idx + 1}/{total_folds_concept}...")

                    # Gap Bagging Split
                    n_items = len(X_final)

                    # Fold 0: Last 20% (Standard) -> Val is [0.8*N : N]
                    # Fold 1: Prev 20% -> Val is [0.6*N : 0.8*N]
                    # Fold 2: Prev 20% -> Val is [0.4*N : 0.6*N]

                    start_pct = 1.0 - 0.2 * (fold_idx + 1)
                    end_pct = 1.0 - 0.2 * fold_idx

                    val_start = int(n_items * start_pct)
                    val_end = int(n_items * end_pct)

                    indices = np.arange(n_items)
                    val_mask = (indices >= val_start) & (indices < val_end)
                    train_mask = ~val_mask

                    X_train, y_train = X_final[train_mask], y_final[train_mask]
                    X_val, y_val = X_final[val_mask], y_final[val_mask]
                    X_test, y_test = X_val, y_val  # Validation IS the test for OOB

                elif force_full_training:
                    current_save_path += ".pth"
                    X_train, y_train = X_final, y_final
                    X_val, y_val = X_final[:2], y_final[:2]
                    X_test, y_test = X_final[:2], y_final[:2]

                    # Check manual epochs
                    model_key_full = f"{asset}_{horizon}_{model_type}"
                    if manual_epochs_dict and model_key_full in manual_epochs_dict:
                        params["epochs_override"] = manual_epochs_dict[model_key_full]
                        print(f"  [Master DL] Forcing Full Training for {params['epochs_override']} Epochs")
                else:
                    current_save_path += ".pth"
                    train_idx = int(len(X_final) * 0.70)
                    val_idx = int(len(X_final) * 0.85)
                    X_train, y_train = X_final[:train_idx], y_final[:train_idx]
                    X_val, y_val = X_final[train_idx:val_idx], y_final[train_idx:val_idx]
                    X_test, y_test = X_final[val_idx:], y_final[val_idx:]

                # Call Helper
                fold_metrics = dl_model._train_model_instance(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    model_type,
                    params,
                    top_features,
                    current_save_path,
                    force_full_training,
                    window_size,
                    manual_epochs=params.get("epochs_override") if force_full_training else None,
                )
                ensemble_metrics.append(fold_metrics)

            # Aggregate Metrics
            avg_metrics = ensemble_metrics[0].copy()
            n_results = len(ensemble_metrics)
            if n_results > 1:
                print(f"  [Ensemble] Aggregating {n_results} folds...")
                for key in avg_metrics:
                    if isinstance(avg_metrics[key], (int, float)):
                        avg_metrics[key] = float(np.mean([m[key] for m in ensemble_metrics]))
                print(
                    f"  [Ensemble] Average F1: {avg_metrics['f1']:.3f} | "
                    f"Acc: {avg_metrics['accuracy']:.3f} | "
                    f"Avg Epochs: {avg_metrics['optimal_epochs']:.1f}",
                )

            results.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "metrics": avg_metrics,
                },
            )

            print(
                f"  Acc: {avg_metrics['accuracy']:.3f} | Precision: {avg_metrics['precision']:.3f} | "
                f"Recall: {avg_metrics['recall']:.3f} | F1: {avg_metrics['f1']:.3f} | AUC: {avg_metrics['auc']:.3f}",
            )

    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    # Save Features Map
    with open(f"{save_dir}/model_features_{model_type}.json", "w", encoding="utf-8") as handle:
        json.dump(model_features_map, handle, indent=4)

    _ = epochs  # kept for API parity with existing call sites
    _ = kwargs
    return results
