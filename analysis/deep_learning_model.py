import os
import json
import torch
from backend.shared.device import configure_torch_runtime, resolve_torch_device
from analysis.dl.model_definitions import (
    FocalLoss,
    LSTMAttentionModel,
    NBeatsNet,
    TimeSeriesDataset,
    TransformerTimeSeriesModel,
)

# Set device
device = resolve_torch_device()
configure_torch_runtime(device)

from config.settings import ENGINEERED_FEATURES_FILE
from analysis.base_model import BaseModel

class DLMacroModel(BaseModel):
    def __init__(self, data_path=ENGINEERED_FEATURES_FILE, model_dir='models_dl'):
        super().__init__(data_path, model_dir)
        self.assets = ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Oil']
        self.horizons = ['1w', '1m', '3m']
        # self.window_size = 60 # Now dynamic per horizon
        self.device = device
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def get_horizon_config(self, horizon, model_type='lstm', asset=None):
        """
        Returns (window_size, model_params) based on horizon and type.
        """
        # Default Baseline
        window_size = 60
        hidden_size = 256
        num_layers = 2
        dropout = 0.3
        
        # Horizon Defaults
        if horizon == '1w':
            window_size = 30
            hidden_size = 128
            num_layers = 2
        elif horizon == '1m':
            window_size = 60
            hidden_size = 256
            num_layers = 2
        elif horizon == '3m':
            window_size = 90
            hidden_size = 256
            num_layers = 3
            dropout = 0.2
            
        params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'model_type': model_type,
            'trans_d_model': 64 if horizon == '1w' else (128 if horizon == '1m' else 256),
            'trans_nhead': 4 if horizon == '1w' else 8,
            'trans_layers': 2 if horizon == '1w' else (3 if horizon == '1m' else 4),
            'trans_dropout': 0.2,
            # N-BEATS Defaults
            'nb_stacks': 2,
            'nb_blocks': 3 if horizon == '1w' else 4,
            'nb_width': 128 if horizon == '1w' else 256
        }
        
        # Override with Optimized Config if available
        config_file = f"{self.model_dir}/dl_config.json"
        
        if os.path.exists(config_file) and asset:
            try:
                with open(config_file, 'r') as f:
                    opt_configs = json.load(f)
                
                key = f"{asset}_{horizon}_{model_type}"
                if key in opt_configs:
                     opt = opt_configs[key]
                     if 'window_size' in opt: window_size = opt['window_size']
                     if 'hidden_size' in opt: params['hidden_size'] = opt['hidden_size']
                     if 'num_layers' in opt: params['num_layers'] = opt['num_layers']
                     if 'dropout' in opt: params['dropout'] = opt['dropout']
                     
                     if 'trans_d_model' in opt: params['trans_d_model'] = opt['trans_d_model']
                     if 'trans_nhead' in opt: params['trans_nhead'] = opt['trans_nhead']
                     if 'trans_layers' in opt: params['trans_layers'] = opt['trans_layers']
                     
                     if 'nb_stacks' in opt: params['nb_stacks'] = opt['nb_stacks']
                     if 'nb_blocks' in opt: params['nb_blocks'] = opt['nb_blocks']
                     if 'nb_width' in opt: params['nb_width'] = opt['nb_width']

                     if 'num_features' in opt: params['num_features'] = opt['num_features']
                     if 'batch_size' in opt: params['batch_size'] = opt['batch_size']
                     if 'focal_gamma' in opt: params['focal_gamma'] = opt['focal_gamma']
            except:
                pass
        
        return window_size, params

            

        

            
    def load_and_preprocess(self):
        """
        Load engineered features with robust fallbacks and ensure required target columns exist.
        """
        from config.settings import ENGINEERED_FEATURES_FILE, MASTER_DATA_FILE
        from analysis.dl.data_pipeline import load_and_preprocess_core

        return load_and_preprocess_core(
            dl_model=self,
            engineered_features_file=ENGINEERED_FEATURES_FILE,
            master_data_file=MASTER_DATA_FILE,
        )

    def create_sequences(self, X_data, window_size, y_data=None, target_alignment='next'):
        """
        Create sliding windows.
        target_alignment:
            - 'next' (default): y[i + window_size] for next-step prediction
            - 'last': y[i + window_size - 1] for same-window alignment
        """
        from analysis.dl.data_pipeline import create_sequences_core

        return create_sequences_core(
            X_data=X_data,
            window_size=window_size,
            y_data=y_data,
            target_alignment=target_alignment,
        )

    def train_all_models(self, model_type='lstm', progress_callback=None, train_cutoff_date=None, epochs=50, config_dict=None, force_full_training=False, manual_epochs_dict=None, use_bagging_ensemble=False, n_folds=3, output_folder=None, **kwargs):
        """
        Train DL models.
        Args:
            force_full_training (bool): If True, train on 100% data.
            manual_epochs_dict (dict): Optimization results {key: epochs}.
            use_bagging_ensemble (bool): If True, trains models (folds) for robustness (Auto-Pilot).
            n_folds (int): Number of folds for ensemble (1 or 3). 1 = Smart Gap Fold (Fast).
            output_folder (str): Custom output folder path (overrides default/holdout logic).
        """
        from analysis.dl.training_pipeline import train_all_models_core

        return train_all_models_core(
            dl_model=self,
            model_type=model_type,
            progress_callback=progress_callback,
            train_cutoff_date=train_cutoff_date,
            epochs=epochs,
            config_dict=config_dict,
            force_full_training=force_full_training,
            manual_epochs_dict=manual_epochs_dict,
            use_bagging_ensemble=use_bagging_ensemble,
            n_folds=n_folds,
            output_folder=output_folder,
            **kwargs,
        )

    def _train_model_instance(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                              model_type, params, top_features, save_path, 
                              force_full_training, window_size, manual_epochs=None, min_epochs=15):
        """
        Helper method to train a single instance (fold) of a model.
        Returns: metrics dictionary
        """
        from analysis.dl.training_core import train_model_instance

        return train_model_instance(
            device=self.device,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type,
            params=params,
            top_features=top_features,
            save_path=save_path,
            force_full_training=force_full_training,
            window_size=window_size,
            manual_epochs=manual_epochs,
            min_epochs=min_epochs,
            lstm_cls=LSTMAttentionModel,
            transformer_cls=TransformerTimeSeriesModel,
            nbeats_cls=NBeatsNet,
            dataset_cls=TimeSeriesDataset,
            loss_cls=FocalLoss,
        )


    def _predict_latest_by_model(self, model_type='lstm'):
        from analysis.dl.inference_engine import predict_latest_by_model

        return predict_latest_by_model(
            dl_model=self,
            model_type=model_type,
            lstm_cls=LSTMAttentionModel,
            transformer_cls=TransformerTimeSeriesModel,
            nbeats_cls=NBeatsNet,
        )

    def predict_latest(self, model_type=None):
        """
        Public latest-prediction API with backward-compatible call styles.
        - predict_latest(model_type='lstm'|'transformer'|'nbeats'|'ensemble')
        - predict_latest() or model_type in {'all', 'range'} for range-based latest snapshot
        """
        if model_type is None:
            return self._predict_latest_from_range()

        normalized = str(model_type).strip().lower()
        if normalized in {'all', 'range'}:
            return self._predict_latest_from_range()
        if normalized not in {'lstm', 'transformer', 'nbeats', 'ensemble'}:
            print(f"Unknown model_type '{model_type}', falling back to range-based latest prediction.")
            return self._predict_latest_from_range()

        return self._predict_latest_by_model(model_type=normalized)

    def predict_ensemble(self):
        """
        Combines predictions from LSTM, Transformer, and N-BEATS using performance-based weighting.
        """
        from analysis.dl.inference_engine import predict_ensemble

        return predict_ensemble(
            dl_model=self,
            lstm_cls=LSTMAttentionModel,
            transformer_cls=TransformerTimeSeriesModel,
            nbeats_cls=NBeatsNet,
        )

    def optimize_models(self, model_type='lstm', epochs=100, batch_size=32, progress_callback=None, save_config=True, base_config=None, train_cutoff_date=None, **kwargs):
        """
        Run Random Search to find best hyperparameters.
        """
        from analysis.dl.hyperparameter_search import optimize_models_core

        return optimize_models_core(
            dl_model=self,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            progress_callback=progress_callback,
            save_config=save_config,
            base_config=base_config,
            train_cutoff_date=train_cutoff_date,
            **kwargs,
        )

    def create_model_snapshot(self, tag=None):
        """
        Creates a backup of the current 'models_dl' folder.
        """
        from analysis.dl.snapshot_store import create_model_snapshot

        return create_model_snapshot(model_dir=self.model_dir, tag=tag)

    def list_model_snapshots(self):
        """
        Returns list of available snapshots.
        """
        from analysis.dl.snapshot_store import list_model_snapshots

        return list_model_snapshots(model_dir=self.model_dir)

    def restore_model_snapshot(self, snapshot_id):
        """
        Restores models from a snapshot (Overwrites current).
        """
        from analysis.dl.snapshot_store import restore_model_snapshot

        return restore_model_snapshot(model_dir=self.model_dir, snapshot_id=snapshot_id)

    def backtest_simple_standard(self, snapshot_id, model_type, asset, initial_capital, start_date, end_date, threshold=0.6):
        """
        Robust Backtesting Engine.
        1. Loads Data & Global Scaler.
        2. Loads Model Configuration & Specific Feature Sets.
        3. Reconstructs EXACT Model Architectures.
        4. Aligns Predictions by Date.
        5. Simulates Trading Strategy.
        """
        from analysis.dl.backtest_engine import backtest_simple_standard_core

        return backtest_simple_standard_core(
            dl_model=self,
            snapshot_id=snapshot_id,
            model_type=model_type,
            asset=asset,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            lstm_cls=LSTMAttentionModel,
            transformer_cls=TransformerTimeSeriesModel,
            nbeats_cls=NBeatsNet,
        )





    def _predict_latest_from_range(self):
        """Generates predictions for the most recent data point."""
        from analysis.dl.range_predictor import predict_latest_from_range_core

        return predict_latest_from_range_core(
            dl_model=self,
            lstm_cls=LSTMAttentionModel,
            transformer_cls=TransformerTimeSeriesModel,
            nbeats_cls=NBeatsNet,
        )

    def predict_range(self, start_date, end_date):
        """Generates predictions for a specific date range using trained DL models."""
        from analysis.dl.range_predictor import predict_range_core

        return predict_range_core(
            dl_model=self,
            start_date=start_date,
            end_date=end_date,
            lstm_cls=LSTMAttentionModel,
            transformer_cls=TransformerTimeSeriesModel,
            nbeats_cls=NBeatsNet,
        )

    # Backtest and Quant Lab Logic Removed by User Request


if __name__ == "__main__":
    dl = DLMacroModel()
    # dl.optimize_models()
    # print(dl.predict_latest())
