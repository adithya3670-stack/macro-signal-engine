import time
import threading
import json
import os
import datetime
import traceback
from typing import Callable, Optional

from backend.services.automation_config_store import AutomationConfigModel, AutomationConfigStore
from backend.services.portfolio_profile_service import PortfolioProfileService

# Lazy imports for heavy modules will happen inside run_pipeline 
# to avoid initial startup lag or circular deps.

class AutomationManager:
    def __init__(
        self,
        data_dir=None,
        http_post: Optional[Callable] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        now_fn: Optional[Callable[[], datetime.datetime]] = None,
        email_service_factory: Optional[Callable[..., object]] = None,
    ):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if data_dir is None:
            self.data_dir = os.path.join(self.base_dir, 'data')
        else:
            self.data_dir = data_dir
            
        self.config_file = os.path.join(self.data_dir, 'automation_config.json')
        self.log_file = os.path.join(self.data_dir, 'automation_log.txt')
        self.config_store = AutomationConfigStore(self.config_file)
        self.config_model: AutomationConfigModel
        
        # ensure data dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.profile_service = PortfolioProfileService(data_dir=self.data_dir)
        
        # State
        self.running = True
        self.is_executing = False
        self.last_run_date = None
        self.status_message = "Idle"
        self.logs = []
        self.last_trigger_key = None
        self.sleep_fn = sleep_fn or time.sleep
        self.now_fn = now_fn or datetime.datetime.now
        self.email_service_factory = email_service_factory
        if http_post is None:
            import requests

            self.http_post = requests.post
        else:
            self.http_post = http_post
        
        # Load Config
        self.config = self._load_config()
        
        # Start Thread
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        
    def _runtime_config_from_model(self, model: AutomationConfigModel):
        payload = model.to_storage_dict()
        payload['_resolved_email_password'] = model.resolved_email_password
        return payload

    def _load_config(self):
        self.config_model = self.config_store.load()
        return self._runtime_config_from_model(self.config_model)

    def _resolved_email_password(self):
        return self.config.get('_resolved_email_password', '')

    def get_public_config(self):
        return self.config_model.to_public_dict()

    def save_config(self, new_config):
        self.config_model = self.config_store.apply_update(self.config_model, new_config)
        self.config = self._runtime_config_from_model(self.config_model)
            
    def log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.logs.append(entry)
        self.status_message = msg
        print(f"[AUTO] {entry}")
        
        # Keep log size managed
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
            
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(entry + "\n")

    def _loop(self):
        # self.log("Scheduler Started.")
        while self.running:
            try:
                if self.config.get("enabled"):
                    now = self.now_fn()
                    target_str = self.config.get("time", "20:30")
                    th, tm = 20, 30
                    
                    try:
                        th, tm = map(int, target_str.split(':'))
                    except:
                        th, tm = 20, 30

                    
                    
                    # SIMPLIFIED LOGIC + DEBOUNCE
                    # Trigger if Hour/Minute matches Target, BUT ensure we haven't triggered for this specific minute yet.
                    
                    current_trigger_key = now.strftime('%Y-%m-%d %H:%M')
                    
                    if now.hour == th and now.minute == tm:
                        # Check if we already triggered for this exact minute
                        if self.last_trigger_key != current_trigger_key:
                            
                            if not self.is_executing:
                                 self.log(f"Triggering Scheduled Run for {current_trigger_key}...")
                            
                            self.last_trigger_key = current_trigger_key  # Mark this minute as handled
                            self.run_pipeline()
                        
                self.sleep_fn(10) # Check every 10s
            except Exception as e:
                print(f"Scheduler Error: {e}")
                self.sleep_fn(60)

    def run_pipeline(self):
        if self.is_executing:
            self.log("Skipping: Pipeline already running.")
            return

        self.is_executing = True
        self.log("Starting Data Pipeline...")
        
        try:
            # Step 1: Update Latest Data
            self.log("Step 1/2: Updating Market Data...")
            response1 = self.http_post('http://127.0.0.1:8002/api/update_latest', timeout=300)
            if response1.status_code == 200:
                result1 = response1.json()
                self.log(f"Market Data Updated: {result1.get('message', 'Success')}")
            else:
                self.log(f"Data Update Failed: {response1.status_code}")
                raise Exception("Data update endpoint failed")
            
            # Step 2: Generate Features  
            self.log("Step 2/2: Generating Features...")
            response2 = self.http_post('http://127.0.0.1:8002/api/update_features', timeout=300)
            if response2.status_code == 200:
                result2 = response2.json()
                self.log(f"Features Generated: {result2.get('message', 'Success')}")
            else:
                self.log(f"Feature Generation Failed: {response2.status_code}")
                raise Exception("Feature generation endpoint failed")
            
            # Pipeline Complete
            self.log("Data Pipeline Completed Successfully.")
            self.save_config({"last_success": self.now_fn().strftime("%Y-%m-%d %H:%M:%S")})
            
            # Check if Automation Lock is enabled
            if self.config.get("lock_enabled") and self.config.get("lock_profile"):
                self.log(f"Automation Lock: Triggering simulation for profile '{self.config['lock_profile']}'...")
                self.run_locked_simulation()
            
        except Exception as e:
            self.log(f"Pipeline Failed: {e}")
            traceback.print_exc()
        finally:
            self.is_executing = False
    
    def run_locked_simulation(self):
        """Run simulation for the locked profile"""
        try:
            profile_name = self.config.get('lock_profile')
            if not profile_name:
                self.log("Automation Lock: No profile specified")
                return
            
            # Load Profile Config
            pf_config = self.profile_service.get_profile(profile_name)
            
            if not pf_config:
                self.log(f"Automation Lock: Profile '{profile_name}' not found.")
                return

            self.log(f"Running simulation for locked profile: {profile_name}")
            
            # Construct API Payload from Saved Profile
            # Note: Saved profile has model_type/dl_folder at top level, but API expects them in strategy_config
            api_payload = {
                'initial_capital': pf_config.get('initial_capital', 10000),
                'start_date': pf_config.get('start_date', '2016-01-01'),
                'universe': pf_config.get('universe', []),
                'custom_cashflows': pf_config.get('custom_cashflows', []),
                'monthly_contribution': pf_config.get('monthly_contribution', 0),
                'trade_threshold': pf_config.get('trade_threshold', 0.5),
                'benchmark_ticker': pf_config.get('benchmark_ticker', 'SP500'),
                'strategy_config': pf_config.get('strategy_config', {}).copy()
            }
            
            # Move model_type and dl_folder into strategy_config where API expects them
            if 'model_type' in pf_config:
                api_payload['strategy_config']['model_type'] = pf_config['model_type']
            if 'dl_folder' in pf_config:
                api_payload['strategy_config']['dl_folder'] = pf_config['dl_folder']

            response = self.http_post(
                'http://127.0.0.1:8002/api/portfolio/run',
                json=api_payload,
                timeout=300
            )
            
            
            if response.status_code == 200:
                result = response.json()
                self.log(f"Automation Lock: Simulation completed for '{profile_name}'")
                
                # Update lock_last_update timestamp
                self.save_config({'lock_last_update': self.now_fn().strftime("%Y-%m-%d %H:%M:%S")})
                
                # --- Send Email Report ---
                if self.config.get('email_enabled') and self.config.get('email_recipient'):
                    try:
                        self.log(f"Sending email report to {self.config['email_recipient']}...")
                        if self.email_service_factory is not None:
                            email_service = self.email_service_factory(
                                sender_email=self.config.get('email_sender', ''),
                                sender_password=self._resolved_email_password(),
                            )
                        else:
                            from services.email_service import EmailService
                            email_service = EmailService(
                            sender_email=self.config.get('email_sender', ''),
                            sender_password=self._resolved_email_password()
                            )
                        
                        # --- Format Data for Email ---
                        # 1. Metrics
                        metrics = {}
                        if 'final_balance' in result:
                            metrics['Final Balance'] = f"${result['final_balance']:,.2f}"
                            
                            # Calculate Total Return
                            # Initial balance is usually the first point in equity curve
                            equity_curve = result.get('equity_curve', [])
                            if equity_curve:
                                initial_bal = equity_curve[0]['Equity']
                                final_bal = result['final_balance']
                                if initial_bal > 0:
                                    total_ret = (final_bal - initial_bal) / initial_bal
                                    metrics['Total Return'] = f"{total_ret:.1%}"
                        
                        raw_metrics = result.get('metrics', {})
                        if 'cagr' in raw_metrics:
                            # Use Total Return if available, else CAGR as proxy? 
                            # Actually, let's calculate Total Return from equity if needed, 
                            # or just show what we have properly formatted.
                            metrics['CAGR'] = f"{raw_metrics['cagr']:.1%}"
                        
                        if 'sharpe' in raw_metrics:
                            metrics['Sharpe'] = f"{raw_metrics['sharpe']:.2f}"
                            
                        if 'max_drawdown' in raw_metrics:
                            metrics['Max DD'] = f"{raw_metrics['max_drawdown']:.1%}"
                            
                        # 2. Holdings (Today's Portfolio)
                        holdings = []
                        latest = result.get('latest_status', {})
                        if latest:
                            # Use held_weights if available
                            weights_dict = latest.get('held_weights', {})
                            total_eq = latest.get('total_equity', 0)
                            latest_sigs = latest.get('latest_signals', {})
                            
                            for asset, w in weights_dict.items():
                                if w > 0.001: # Filter tiny dust
                                    val = w * total_eq
                                    sig = latest_sigs.get(asset, 0)
                                    holdings.append({
                                        'Asset': asset,
                                        'Weight': f"{w:.1%}",
                                        'Value': f"${val:,.0f}",
                                        'Signal': f"{sig:.0%}"
                                    })
                        
                        # 3. Transactions (Last 5)
                        transactions = []
                        raw_trades = result.get('trades', [])
                        if raw_trades:
                            #Sort by date desc just in case
                            raw_trades.sort(key=lambda x: x.get('Date', ''), reverse=True)
                            for t in raw_trades[:5]:
                                transactions.append({
                                    'Date': t.get('Date', ''),
                                    'Asset': t.get('Asset', ''),
                                    'Action': t.get('Action', '').upper(),
                                    'Price': f"${t.get('Price', 0):,.2f}",
                                    'Size': f"{t.get('Weight', 0):.4f}" # Use Weight or Size if available? Trades usually have Weight
                                })

                        # 4. Actions Required Today (Contributions + Rebalancing)
                        actions = []
                        contribution_amount = 0
                        if latest:
                            held = latest.get('held_weights', {})
                            targets = latest.get('target_weights', {})
                            total_eq = latest.get('total_equity', 0)
                            
                            # Check for today's trades (contributions or rebalancing)
                            today_date = latest.get('date', '')
                            if raw_trades and today_date:
                                today_trades = [t for t in raw_trades if t.get('Date') == today_date]
                                
                                # Show all of today's trades as "Actions Required Today"
                                for t in today_trades:
                                    asset = t.get('Asset', '')
                                    action = t.get('Action', '').upper()
                                    value = t.get('Value', 0)
                                    weight = t.get('Weight', 0)
                                    
                                    if value > 0 and asset != 'Cash':
                                        actions.append({
                                            'asset': asset,
                                            'action': action,
                                            'amount': f"${value:,.0f}",
                                            'details': f"{weight:.1%} allocation"
                                        })
                                
                                # Calculate total contribution (sum of BUY values)
                                today_buys = [t for t in today_trades if t.get('Action', '').upper() == 'BUY']
                                if today_buys:
                                    contribution_amount = sum(t.get('Value', 0) for t in today_buys)
                            
                            # Only check for rebalancing if no trades happened today
                            if not actions:
                                # Rebalancing actions (weight differences)
                                trade_threshold = pf_config.get('trade_threshold', 0.5) / 100
                                all_assets = set(list(held.keys()) + list(targets.keys()))
                                for asset in all_assets:
                                    if asset == 'Cash':
                                        continue
                                    held_wt = held.get(asset, 0)
                                    target_wt = targets.get(asset, 0)
                                    delta = target_wt - held_wt
                                    
                                    if abs(delta) > trade_threshold:
                                        current_value = held_wt * total_eq
                                        target_value = target_wt * total_eq
                                        dollar_change = abs(target_value - current_value)
                                        
                                        if delta > 0:
                                            actions.append({
                                                'asset': asset,
                                                'action': 'BUY',
                                                'amount': f"${dollar_change:,.0f}",
                                                'details': f"{held_wt:.1%} → {target_wt:.1%}"
                                            })
                                        else:
                                            actions.append({
                                                'asset': asset,
                                                'action': 'SELL',
                                                'amount': f"${dollar_change:,.0f}",
                                                'details': f"{held_wt:.1%} → {target_wt:.1%}"
                                            })

                        formatted_data = {
                            'metrics': metrics,
                            'holdings': holdings,
                            'transactions': transactions,
                            'actions': actions,
                            'contribution_amount': f"${contribution_amount:,.0f}" if contribution_amount > 0 else None
                        }

                        success, msg = email_service.send_portfolio_report(
                            recipient_email=self.config['email_recipient'],
                            profile_name=profile_name,
                            portfolio_data=formatted_data
                        )
                        
                        if success:
                            self.log("Email report sent successfully.")
                        else:
                            self.log(f"Failed to send email: {msg}")
                            
                    except Exception as email_err:
                        self.log(f"Email error: {email_err}")
                        traceback.print_exc()
            else:
                self.log(f"Automation Lock: Simulation failed with status {response.status_code}")
                try:
                    error_detail = response.json()
                    self.log(f"Error details: {error_detail}")
                except:
                    self.log(f"Error response: {response.text[:500]}")

                
        except Exception as e:
            self.log(f"Automation Lock: Simulation error - {e}")
            traceback.print_exc()
    
    def get_lock_config(self):
        """Get current lock configuration"""
        return {
            'lock_enabled': self.config.get('lock_enabled', False),
            'lock_profile': self.config.get('lock_profile', ''),
            'lock_last_update': self.config.get('lock_last_update'),
            'email_enabled': self.config.get('email_enabled', False),
            'email_recipient': self.config.get('email_recipient', ''),
            'email_sender': self.config.get('email_sender', ''),
            'email_password': '*****' if self._resolved_email_password() else '',
            'email_password_env': self.config.get('email_password_env', 'MACRO_AUTO_EMAIL_PASSWORD'),
        }
    
    def save_lock_config(self, lock_data):
        """Save lock configuration"""
        self.save_config(lock_data)
