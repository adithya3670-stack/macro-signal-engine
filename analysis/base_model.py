from abc import ABC, abstractmethod
import os
import pandas as pd
from config.settings import ENGINEERED_FEATURES_FILE

class BaseModel(ABC):
    """
    Abstract Base Class for all Macro Economic Models.
    Enforces a standard interface for training and inference.
    """
    def __init__(self, data_path=ENGINEERED_FEATURES_FILE, model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    @abstractmethod
    def train_all_models(self, progress_callback=None, **kwargs):
        """
        Train models for all assets.
        :param progress_callback: Function to call with (progress_pct, message)
        :param kwargs: Model-specific arguments (e.g., epochs, model_type)
        """
        pass

    @abstractmethod
    def predict_latest(self):
        """
        Generate predictions for the most recent available data point.
        :return: Dictionary of {asset: prediction_score}
        """
        pass
        
    def load_data(self):
        """
        Common data loading utility.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        return df.sort_index()
