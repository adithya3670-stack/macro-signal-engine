from flask import Flask, render_template
import logging

# --- Blueprints ---
from analysis.sentiment_analysis import sentiment_bp
from analysis.backtest_api import backtest_bp
from routes.backtest_holdout import holdout_bp
from routes.automation import automation_bp
from routes.analysis import analysis_bp
from routes.training import training_bp
from routes.ensembling import ensembling_bp
from routes.combination_optimizer import combo_bp
from routes.sequential_training import sequential_bp

# Filter out /api/automation/logs from werkzeug logger to reduce noise
class NoPollLogs(logging.Filter):
    def filter(self, record):
        return '/api/automation/logs' not in record.getMessage()

# Get werkzeug logger and add filter
log = logging.getLogger('werkzeug')
log.addFilter(NoPollLogs())

app = Flask(__name__)

# --- Register Blueprints ---
app.register_blueprint(sentiment_bp, url_prefix='/api/sentiment')
app.register_blueprint(backtest_bp)  # Legacy Backtest Routes (keep for now)
app.register_blueprint(holdout_bp)   # New Holdout/Backtest Logic
app.register_blueprint(automation_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(training_bp)
app.register_blueprint(ensembling_bp)  # Model Ensembling Tab
app.register_blueprint(combo_bp)       # Combination Efficiency Tab
app.register_blueprint(sequential_bp)   # Sequential Training Tab

@app.route('/')
def index():
    return render_template('index.html')

# Force browser to never cache - always fetch fresh content
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    # use_reloader=False prevents the server from restarting when files change (e.g. models training)
    # This ensures "Connection Interrupted" errors don't happen due to watchdog restarts.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8002)
