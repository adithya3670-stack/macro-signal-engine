import os

# Base Directory (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(DATA_DIR, 'logs')

# File Names
MASTER_DATA_FILE = os.path.join(BASE_DIR, 'master_data', 'master_dataset.csv')
ENGINEERED_FEATURES_FILE = os.path.join(DATA_DIR, 'engineered_features.csv')
PRICE_3D_FEATURES_FILE = os.path.join(DATA_DIR, 'price_3d_features.csv')
PRICE_1W_FEATURES_FILE = os.path.join(DATA_DIR, 'price_1w_features.csv')
PRICE_1M_FEATURES_FILE = os.path.join(DATA_DIR, 'price_1m_features.csv')
EXTRACTED_DATA_FILE = os.path.join(DATA_DIR, 'extracted_data.csv')
SP500_DRAWDOWN_EVENTS_FILE = os.path.join(DATA_DIR, 'sp500_drawdown_events_classified.csv')
RESEARCH_MINOR_EVENT_DIR = os.path.join(DATA_DIR, 'research_minor_event_signals')
MINOR_3D_RESEARCH_SCRIPT = os.path.join(BASE_DIR, 'scripts', 'research_minor_3d_precision.py')

# Raw Data Files
MARKET_DATA_FILE = os.path.join(DATA_DIR, 'market_data.csv')
MACRO_DATA_FILE = os.path.join(DATA_DIR, 'macro_data.csv')
SENTIMENT_DATA_FILE = os.path.join(DATA_DIR, 'sentiment_data.csv')
INDICATORS_DATA_FILE = os.path.join(DATA_DIR, 'indicators_data.csv')
COMMODITIES_DATA_FILE = os.path.join(DATA_DIR, 'commodities_data.csv')

# Model Paths
HOLDOUT_DIR = os.path.join(MODELS_DIR, 'holdout')
HOLDOUT_DL_DIR = os.path.join(MODELS_DIR, 'holdout_dl')
HOLDOUT_PRICE_3D_DIR = os.path.join(MODELS_DIR, 'holdout_price_3d')
HOLDOUT_PRICE_1W_DIR = os.path.join(MODELS_DIR, 'holdout_price_1w')
HOLDOUT_PRICE_1M_DIR = os.path.join(MODELS_DIR, 'holdout_price_1m')
MODELS_PRICE_3D_DIR = os.path.join(BASE_DIR, 'models_price_3d')
MODELS_PRICE_1W_DIR = os.path.join(BASE_DIR, 'models_price_1w')
MODELS_PRICE_1M_DIR = os.path.join(BASE_DIR, 'models_price_1m')
MODELS_REGIME_DIR = os.path.join(BASE_DIR, 'models_regime')
METRICS_FILE = os.path.join(MODELS_DIR, 'model_metrics.json')

# Assets
ASSETS = ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Oil']

# Ensure Directories Exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HOLDOUT_PRICE_3D_DIR, exist_ok=True)
os.makedirs(HOLDOUT_PRICE_1W_DIR, exist_ok=True)
os.makedirs(HOLDOUT_PRICE_1M_DIR, exist_ok=True)
os.makedirs(MODELS_PRICE_3D_DIR, exist_ok=True)
os.makedirs(MODELS_PRICE_1W_DIR, exist_ok=True)
os.makedirs(MODELS_PRICE_1M_DIR, exist_ok=True)
os.makedirs(MODELS_REGIME_DIR, exist_ok=True)
os.makedirs(RESEARCH_MINOR_EVENT_DIR, exist_ok=True)
