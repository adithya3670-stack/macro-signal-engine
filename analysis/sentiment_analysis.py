from flask import Blueprint, jsonify, request
from data.sentiment_data import get_sentiment_data
import pandas as pd
from datetime import datetime, timedelta

sentiment_bp = Blueprint('sentiment_analysis', __name__)

@sentiment_bp.route('/data')
def get_sentiment():
    try:
        # Default to 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        df = get_sentiment_data(start_date, end_date)
        
        # Format for JSON
        df = df.reset_index()
        # Handle date column name
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
        df.rename(columns={date_col: 'Date'}, inplace=True)
        
        return jsonify({
            'dates': df['Date'].tolist(),
            'vix': df['VIX'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
