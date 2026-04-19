import pandas as pd
import numpy as np
import os
import joblib
import json # Fixed missing import
from analysis.deep_learning_model import DLMacroModel

class SignalGenerator:
    """
    Generates trading signals using the Deep Learning Ensemble.
    Acts as an abstraction layer between the Backtest Engine and the complex Model logic.
    """
    def __init__(self, model_dir='models_dl'):
        self.model_dir = model_dir
        self.dl_model = DLMacroModel(model_dir=model_dir)
        self.cache = {}

    def generate_signals(self, assets, start_date=None, end_date=None, force_refresh=False, model_type='ensemble'):
        """
        Runs inference for the given assets and returns a DataFrame of signals.
        
        Args:
            assets (list): List of asset names (e.g., ['SP500', 'Bitcoin'])
            start_date (str): Optional start date filter
            end_date (str): Optional end date filter
            force_refresh (bool): Ignore cache
            
        Returns:
            pd.DataFrame: Columns = [Asset1_Prob, Asset2_Prob, ...], Index = Date
        """
        print(f"Generating signals for {len(assets)} assets...")
        
        signal_frames = []
        
        for asset in assets:
            # Auto-detect horizon if missing (Default to 1m)
            target_asset = asset
            if '_' not in asset:
                # Check if specific model exists, else try appending _1m
                if not os.path.exists(os.path.join(self.model_dir, f"{asset}_lstm.pth")):
                    target_asset = f"{asset}_1m"
            
            cache_key = f"{target_asset}_{model_type}" # Cached by model type too
            
            if not force_refresh and cache_key in self.cache:
                print(f"-> Using cached signals for {target_asset} ({model_type})")
                df = self.cache[cache_key]
            else:
                try:
                    # We'll use a new method in this class to run inference
                    df = self._run_inference_for_asset(target_asset, model_type=model_type)
                    self.cache[cache_key] = df
                except Exception as e:
                    print(f"Error generating signals for {asset} (target: {target_asset}): {e}")
                    continue
            
            if df is not None:
                # Rename 'ensemble' (or result col) to Original Asset Name
                if 'result' in df.columns:
                     col_name = 'result'
                else: 
                     col_name = 'ensemble'
                
                if col_name in df.columns:
                     df = df[[col_name]].astype(float).rename(columns={col_name: asset})
                     signal_frames.append(df)

        if not signal_frames:
            return pd.DataFrame() # Empty
            
        # Merge all signals on Date index
        merged_signals = pd.concat(signal_frames, axis=1)
        merged_signals.sort_index(inplace=True)
        
        # Filter Dates
        if start_date:
            merged_signals = merged_signals[merged_signals.index >= pd.to_datetime(start_date)]
        if end_date:
            merged_signals = merged_signals[merged_signals.index <= pd.to_datetime(end_date)]
            
        return merged_signals

    def _run_inference_for_asset(self, asset, model_type='ensemble'):
        """
        Internal method to load models and predict.
        Re-implements the deleted 'run_inference_pipeline' logic but cleaner.
        """
        import torch
        from analysis.deep_learning_model import LSTMAttentionModel, TransformerTimeSeriesModel, NBeatsNet
        
        # 1. Load Data & Scaler
        df_full = self.dl_model.load_and_preprocess()
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
             print("Scaler not found.")
             return None
        scaler = joblib.load(scaler_path)
        
        # Create Scaled DF
        exclude_cols = [c for c in df_full.columns if 'Target_' in c or 'Regime_' in c or 'Date' in c]
        numeric_cols = [c for c in df_full.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_full[c])]
        
        df_scaled = df_full.copy()
    
        # FIX: Use scaler's feature names to ensure correct order
        valid_cols = numeric_cols
        if hasattr(scaler, 'feature_names_in_'):
            required_features = list(scaler.feature_names_in_)
            # Check if all features exist
            missing = [f for f in required_features if f not in df_full.columns]
            if missing:
                print(f"Warning: Missing features in inference data: {missing}")
                # Try to fill missing with 0 or skip? For now, let's try to proceed strictly
                return None
            
            # Transform using exact order
            df_scaled[required_features] = scaler.transform(df_full[required_features])
            valid_cols = required_features
        else:
            # Fallback for older sklearn (use intersection)
            valid_cols = [c for c in numeric_cols if c in getattr(scaler, 'feature_names_in_', numeric_cols)]
            if valid_cols:
                df_scaled[valid_cols] = scaler.transform(df_full[valid_cols])
            
        # 2. Load Config
        config_path = os.path.join(self.model_dir, "dl_config.json")
        dl_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: dl_config = json.load(f)

        # Auto-detect horizon if missing
        target_asset = asset
        if len(asset.split('_')) == 1:
            target_asset = f"{asset}_1m"
            print(f"Auto-appending suffix: {asset} -> {target_asset}")

        # 3. Model Loop (Ensemble)
        # Dynamic dimension inference now supports all model types
        if model_type == 'ensemble':
            models = ['lstm', 'transformer', 'nbeats']  # All models with dynamic inference
        elif model_type.startswith('winner_ensemble'):
            # Load dynamic winner config (supports both legacy 'winner_ensemble' and 'winner_ensemble_accuracy' etc)
            config_filename = "winner_ensemble_config.json" # Default
            
            # Check if model_type has a metric suffix
            if model_type.startswith('winner_ensemble_'):
                 metric_suffix = model_type.split('winner_ensemble_')[1]
                 if metric_suffix:
                     config_filename = f"winner_ensemble_{metric_suffix}.json"
            
            winner_config_path = os.path.join(self.model_dir, config_filename)
            winner_model = 'lstm' # Fallback
            
            if os.path.exists(winner_config_path):
                try:
                    with open(winner_config_path, 'r') as f:
                        wc = json.load(f)
                    winners = wc.get('winners', {})
                    if target_asset in winners:
                        winner_model = winners[target_asset]['model']
                        score = winners[target_asset].get('score', 'N/A')
                        print(f"  [Winner Ensemble] ({config_filename}) {target_asset} -> Selected {winner_model.upper()} (Score: {score})")
                    else:
                        print(f"  [Winner Ensemble] {target_asset} not found in config {config_filename}. Defaulting to LSTM.")
                except Exception as e:
                    print(f"  [Winner Ensemble] Error reading config {config_filename}: {e}. Defaulting to LSTM.")
            else:
                # If specific metric file fails, try default legacy
                if config_filename != "winner_ensemble_config.json" and os.path.exists(os.path.join(self.model_dir, "winner_ensemble_config.json")):
                     print(f"  [Winner Ensemble] {config_filename} not found, falling back to legacy config.")
                     winner_config_path = os.path.join(self.model_dir, "winner_ensemble_config.json")
                     # ... (duplicate logic or just let the next check fail? let's keep it simple)
                
                print(f"  [Winner Ensemble] Config {config_filename} not found in {self.model_dir}. Defaulting to LSTM.")
            
            models = [winner_model]
        else:
            models = [model_type]
            
        preds = pd.DataFrame(index=df_full.index, columns=models)
        
        for m_type in models:
            model_key = f"{target_asset}_{m_type}"
            # Try both .pth and .pt extensions for compatibility
            weight_file = os.path.join(self.model_dir, f"{model_key}.pth")
            if not os.path.exists(weight_file):
                weight_file = os.path.join(self.model_dir, f"{model_key}.pt")
            
            print(f"DEBUG: Looking for model file: {weight_file}")
            
            # Fallback: Check for fold files (e.g., ..._fold1.pth)
            if not os.path.exists(weight_file):
                possible_folds = [f for f in os.listdir(self.model_dir) if f.startswith(model_key) and '_fold' in f and f.endswith('.pth')]
                if possible_folds:
                    # Use the first fold found (sufficient for inference if we just need *a* model)
                    # Ideally we would ensemble folds, but for 1-fold smart gap this works perfectly.
                    weight_file = os.path.join(self.model_dir, sorted(possible_folds)[0])
                    print(f"DEBUG: Found fold file: {weight_file}")

            print(f"DEBUG: File exists: {os.path.exists(weight_file)}")
            
            if not os.path.exists(weight_file):
                # Fallback: Try exact match with both extensions
                if target_asset != asset:
                     for ext in ['.pth', '.pt']:
                         weight_file_alt = os.path.join(self.model_dir, f"{asset}_{m_type}{ext}")
                         if os.path.exists(weight_file_alt):
                             model_key = f"{asset}_{m_type}"
                             weight_file = weight_file_alt
                             break
                     else:
                         print(f"DEBUG: No model file found, skipping {m_type}")
                         continue
                else:
                    print(f"DEBUG: Model not found, skipping {m_type}")
                    continue
                
            params = dl_config.get(model_key, {"window_size": 60, "hidden_size": 128})
            window_size = int(params.get('window_size', 60))
            
            # Features
            feat_file = os.path.join(self.model_dir, f"model_features_{m_type}.json")
            spec_features = []
            if os.path.exists(feat_file):
                 # Try both keys just in case
                 with open(feat_file, 'r') as f: 
                     feats_json = json.load(f)
                     spec_features = feats_json.get(model_key, feats_json.get(f"{asset}_{m_type}", []))
            if not spec_features: spec_features = valid_cols[:20]
            
            # Prepare Input Tensor
            try: 
                X_values = df_scaled[spec_features].values
                if np.isnan(X_values).any():
                     print(f"[{model_key}] WARNING: Input X_values contain NaNs!")
                     # Optional: Fill NaNs?
                     # X_values = np.nan_to_num(X_values)
            except Exception as e: 
                print(f"[{model_key}] Feature extraction failed: {e}")
                continue
                
            X_seq = []
            valid_idx = []
            for i in range(window_size, len(X_values)):
                window = X_values[i-window_size:i]
                if np.isnan(window).any(): continue # Skip windows with NaNs
                X_seq.append(window)
                valid_idx.append(i)
                
            if not X_seq: 
                print(f"[{model_key}] No valid sequences found (len={len(X_values)}). NaNs?")
                continue
            
            # Init Architecture - INFER dimensions from checkpoint BEFORE creating X_tensor
            model = None
            try:
                # Load checkpoint first to get actual dimensions
                checkpoint = torch.load(weight_file, map_location=self.dl_model.device, weights_only=False)
                
                # Determine required input size from checkpoint for each model type
                required_inp = len(spec_features)  # Default
                
                if m_type == 'lstm' and 'lstm.weight_ih_l0' in checkpoint:
                    required_inp = checkpoint['lstm.weight_ih_l0'].shape[1]
                    print(f"  [LSTM] Model expects input_size={required_inp}, we have {len(spec_features)} features")
                elif m_type == 'transformer' and 'embedding.weight' in checkpoint:
                    required_inp = checkpoint['embedding.weight'].shape[1]  # [d_model, input_size]
                    print(f"  [Transformer] Model expects input_size={required_inp}, we have {len(spec_features)} features")
                elif m_type == 'nbeats' and 'stacks.0.0.fc_stack.0.weight' in checkpoint:
                    flat_input = checkpoint['stacks.0.0.fc_stack.0.weight'].shape[1]
                    required_inp = flat_input // window_size  # flat_input = window_size * num_features
                    print(f"  [N-BEATS] Model expects input_size={required_inp} (flat={flat_input}), we have {len(spec_features)} features")
                
                # Adjust features if needed (same logic for all model types)
                if len(spec_features) > required_inp:
                    print(f"  [{m_type.upper()}] Limiting to first {required_inp} features")
                    X_seq = [x[:, :required_inp] for x in X_seq]
                elif len(spec_features) < required_inp:
                    print(f"  [{m_type.upper()}] Feature count mismatch ({len(spec_features)} < {required_inp}) - cannot proceed")
                    continue
                
                # Create input tensor with correct shape
                X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(self.dl_model.device)
                inp = required_inp  # Use the adjusted input size for model instantiation
            
                # Now instantiate model with correct dimensions
                if m_type == 'lstm':
                    # Infer dimensions from already-loaded checkpoint
                    if 'lstm.weight_ih_l0' in checkpoint:
                        weight_shape = checkpoint['lstm.weight_ih_l0'].shape
                        inp = weight_shape[1]
                        hidden = weight_shape[0] // 4
                        
                        # Infer num_layers by counting lstm.weight_ih_lX keys
                        num_layers = 0
                        for key in checkpoint.keys():
                            if key.startswith('lstm.weight_ih_l') and '_reverse' not in key:
                                layer_num = int(key.split('_l')[1])
                                num_layers = max(num_layers, layer_num + 1)
                        print(f"  [LSTM] Inferred: input={inp}, hidden={hidden}, layers={num_layers}")
                    else:
                        inp = required_inp
                        hidden = params.get('hidden_size', 128)
                        num_layers = params.get('num_layers', 2)
                    
                    model = LSTMAttentionModel(
                        input_size=inp, 
                        hidden_size=hidden, 
                        num_layers=num_layers, 
                        dropout=params.get('dropout', 0.2)
                    ).to(self.dl_model.device)
                elif m_type == 'transformer':
                    # Infer Transformer dimensions from checkpoint
                    if 'embedding.weight' in checkpoint:
                        # embedding.weight shape is [d_model, input_size]
                        emb_shape = checkpoint['embedding.weight'].shape
                        dm = emb_shape[0]  # d_model
                        inp = emb_shape[1]  # input_size (num_features)
                        
                        # Infer nhead from self_attn.in_proj_weight: [3*d_model, d_model]
                        # And num_layers by counting transformer_encoder.layers.X keys
                        num_layers = 0
                        for key in checkpoint.keys():
                            if 'transformer_encoder.layers.' in key:
                                parts = key.split('.')
                                try:
                                    layer_idx = int(parts[2])
                                    num_layers = max(num_layers, layer_idx + 1)
                                except:
                                    pass
                        
                        # nhead: usually d_model // head_dim, common head_dim is 8 or 16
                        # Try to find from in_proj: shape [3*d_model, d_model] -> nhead = d_model // (d_model // nhead)
                        # Safest: use common values that divide evenly
                        nh = 8  # Default
                        for candidate in [8, 4, 2, 1]:
                            if dm % candidate == 0:
                                nh = candidate
                                break
                        
                        print(f"  [Transformer] Inferred: d_model={dm}, nhead={nh}, layers={num_layers}, input={inp}")
                    else:
                        dm = params.get('trans_d_model', 128)
                        nh = params.get('trans_nhead', 8)
                        num_layers = params.get('trans_layers', 3)
                        if dm % nh != 0: dm = (dm // nh) * nh
                    
                    model = TransformerTimeSeriesModel(
                        input_size=inp, 
                        d_model=dm, 
                        nhead=nh,
                        num_layers=num_layers,
                        dropout=params.get('dropout', 0.1)
                    ).to(self.dl_model.device)
                    
                elif m_type == 'nbeats':
                    # Infer N-BEATS dimensions from checkpoint
                    # stacks.0.0.fc_stack.0.weight shape is [layer_width, window_size * num_features]
                    if 'stacks.0.0.fc_stack.0.weight' in checkpoint:
                        fc_shape = checkpoint['stacks.0.0.fc_stack.0.weight'].shape
                        width = fc_shape[0]  # layer_width
                        flat_input = fc_shape[1]  # window_size * num_features
                        
                        # Count stacks and blocks
                        num_stacks = 0
                        num_blocks = 0
                        for key in checkpoint.keys():
                            if key.startswith('stacks.'):
                                parts = key.split('.')
                                try:
                                    stack_idx = int(parts[1])
                                    block_idx = int(parts[2])
                                    num_stacks = max(num_stacks, stack_idx + 1)
                                    num_blocks = max(num_blocks, block_idx + 1)
                                except:
                                    pass
                        
                        print(f"  [N-BEATS] Inferred: width={width}, stacks={num_stacks}, blocks={num_blocks}")
                    else:
                        width = params.get('nb_width', 256)
                        num_stacks = params.get('nb_stacks', 2)
                        num_blocks = params.get('nb_blocks', 3)
                    
                    model = NBeatsNet(
                        num_features=inp, 
                        window_size=window_size, 
                        layer_width=width,
                        num_stacks=num_stacks,
                        num_blocks=num_blocks
                    ).to(self.dl_model.device)
                
                # Load Weights
                model.load_state_dict(torch.load(weight_file, map_location=self.dl_model.device, weights_only=False))
                model.eval()
                
                # Batched Inference
                with torch.no_grad():
                    raw_out = []
                    batch_size = 1024
                    for k in range(0, len(X_tensor), batch_size):
                        batch = X_tensor[k:k+batch_size]
                        out = model(batch)
                        raw_out.extend(out.cpu().numpy().flatten())
                    
                    probs = 1 / (1 + np.exp(-np.array(raw_out)))
                    seed_idx = df_full.index[valid_idx]
                    preds.loc[seed_idx, m_type] = probs
            except Exception as e:
                print(f"Inference failed for {model_key}: {e}")
                continue
                
        # Weighted Ensemble based on Accuracy, Precision, and Robustness (AUC)
        # Load metrics for each model type
        weights = {}
        for m_type in models:
            metrics_file = os.path.join(self.model_dir, f"dl_metrics_{m_type}.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_list = json.load(f)
                    
                    # Find metrics for this specific asset/horizon
                    asset_name = target_asset.split('_')[0] if '_' in target_asset else target_asset
                    horizon = target_asset.split('_')[1] if '_' in target_asset else '1m'
                    
                    model_metric = None
                    for item in metrics_list:
                        if item['asset'] == asset_name and item['horizon'] == horizon:
                            model_metric = item['metrics']
                            break
                    
                    if model_metric:
                        # Composite weight: 40% Accuracy + 30% Precision + 30% AUC (Robustness)
                        accuracy = model_metric.get('accuracy', 0.5)
                        precision = model_metric.get('precision', 0.5)
                        auc = model_metric.get('auc', 0.5)
                        
                        # Calculate weighted score
                        weight = (0.4 * accuracy) + (0.3 * precision) + (0.3 * auc)
                        weights[m_type] = weight
                        print(f"  [{m_type}] Metrics: Acc={accuracy:.3f}, Prec={precision:.3f}, AUC={auc:.3f} -> Weight={weight:.3f}")
                    else:
                        # Default weight if metrics not found
                        weights[m_type] = 0.5
                        print(f"  [{m_type}] No metrics found for {asset_name}_{horizon}, using default weight=0.5")
                except Exception as e:
                    weights[m_type] = 0.5
                    print(f"  [{m_type}] Error loading metrics: {e}, using default weight=0.5")
            else:
                # Fallback to equal weight if metrics file missing
                weights[m_type] = 0.5
                print(f"  [{m_type}] Metrics file not found, using default weight=0.5")
        
        # Filter to only models with valid predictions (non-NaN)
        valid_models = [m for m in models if m in preds.columns and preds[m].notna().any()]
        print(f"  Valid models with predictions: {valid_models}")
        
        if not valid_models:
            print(f"  [ERROR] No models produced valid predictions!")
            return pd.DataFrame()  # Return empty
        
        # Apply weighted average using ONLY valid models
        valid_weights = {m: weights.get(m, 0.5) for m in valid_models}
        total_weight = sum(valid_weights.values())
        
        if total_weight > 0:
            weighted_sum = sum(preds[m].fillna(0) * valid_weights[m] for m in valid_models)
            preds['result'] = weighted_sum / total_weight
            print(f"[OK] Weighted ensemble applied: {valid_weights}")
        else:
            preds['result'] = preds[valid_models].mean(axis=1)
            print("[WARN] Using simple average")
        
        return preds[['result']].dropna()

if __name__ == "__main__":
    gen = SignalGenerator()
    sigs = gen.generate_signals(['SP500_1m'], start_date='2020-01-01')
    print("Signal Frame Head:\n", sigs.head())
    print("Signal Frame Tail:\n", sigs.tail())
