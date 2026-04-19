from flask import Blueprint, jsonify, request
import threading
from analysis.automation_manager import AutomationManager

automation_bp = Blueprint('automation', __name__)

# Singleton Automation Manager
auto_manager = AutomationManager()

@automation_bp.route('/api/automation/config', methods=['GET', 'POST'])
def api_automation_config():
    if request.method == 'POST':
        data = request.json
        auto_manager.save_config(data)
        return jsonify({'status': 'success', 'config': auto_manager.get_public_config()})
    return jsonify(auto_manager.get_public_config())

@automation_bp.route('/api/automation/logs', methods=['GET'])
def api_automation_logs():
    return jsonify({
        'logs': auto_manager.logs,
        'status': auto_manager.status_message,
        'is_running': auto_manager.is_executing
    })

@automation_bp.route('/api/automation/run_now', methods=['POST'])
def api_automation_run():
    # Run in background thread to not block
    t = threading.Thread(target=auto_manager.run_pipeline)
    t.start()
    return jsonify({'status': 'success', 'message': 'Pipeline triggered.'})

@automation_bp.route('/api/automation/lock', methods=['GET', 'POST'])
def api_automation_lock():
    """Get or save automation lock configuration"""
    if request.method == 'POST':
        data = request.json
        auto_manager.save_lock_config(data)
        return jsonify({
            'status': 'success',
            'config': auto_manager.get_lock_config()
        })
    return jsonify(auto_manager.get_lock_config())
