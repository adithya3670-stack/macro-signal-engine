import pandas as pd
import numpy as np

class VectorizedBacktester:
    """
    High-Performance Vectorized Backtester.
    Calculates portfolio returns based on weights and price changes.
    """
    def __init__(self, initial_capital=10000.0, transaction_cost_bps=10.0):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10000.0 # e.g. 0.0010

    def _calculate_safe_returns(self, prices):
        """
        Calculates returns robustly, handling negative prices (e.g., Oil 2020).
        Formula: (P_t - P_{t-1}) / |P_{t-1}|
        This preserves directionality when price is negative.
        """
        diff = prices.diff()
        shifted = prices.shift(1)
        # Use abs(shifted) to ensure sign of return matches sign of price movement
        returns = diff / shifted.abs()
        # Handle division by zero or infs
        return returns.fillna(0).replace([np.inf, -np.inf], 0)

    def run_backtest(self, prices_df, weights_df, trade_threshold=0.005):
        """
        Core Simulation Loop (Vectorized).
        
        Args:
            prices_df (pd.DataFrame): Asset Close prices.
            weights_df (pd.DataFrame): Target allocation weights (0 to 1).
            trade_threshold (float): Minimum weight change to log a trade (0.01 = 1%).
            
        Returns:
            dict: { 'metrics': {}, 'equity_curve': pd.Series }
        """
        # 1. Calculate Asset Returns
        # 1. Calculate Asset Returns
        # Use safe returns to handle negative prices
        asset_returns = self._calculate_safe_returns(prices_df)
        
        # 2. Align Weights (Lag by 1 period)
        # We trade at Close of T, to get Return of T+1.
        # So Weights calculated at T apply to Returns of T+1.
        # shift(1) moves T to T+1 row.
        aligned_weights = weights_df.shift(1).fillna(0)
        aligned_weights = aligned_weights.reindex(asset_returns.index).fillna(0)
        
        # 3. Calculate Portfolio Return
        # Sum (Weight * Asset_Ret) across columns
        # shape: (n_days, n_assets)
        weighted_returns = asset_returns * aligned_weights
        portfolio_returns = weighted_returns.sum(axis=1)
        
        # 4. Transaction Costs
        # Turnover = abs(Weight_t - Weight_{t-1})
        turnover = aligned_weights.diff().abs().sum(axis=1).fillna(0)
        t_costs = turnover * self.transaction_cost
        
        net_returns = portfolio_returns - t_costs
        
        # 5. Equity Curve
        # (1 + r).cumprod()
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()
        
        # 6. Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        metrics = self._calculate_metrics(net_returns, drawdown)
        
        return {
            'equity_curve': equity_curve,
            'drawdown': drawdown,
            'weights': aligned_weights,
            'metrics': metrics,
            'daily_returns': net_returns,
            'trades': self._generate_trade_log(weights_df, prices_df, equity_curve, trade_threshold)
        }

    def _generate_trade_log(self, weights, prices, equity, threshold=0.005):
        """Generates a list of trades based on weight changes."""
        trades = []
        # Calculate weight differences (Turnover)
        # weights index is the rebalance date (Close)
        diffs = weights.diff().fillna(0)
        
        # Handle initial entry (First non-zero weights)
        # diff().fillna(0) treats first row as NaN->0, so it misses the initial buy.
        # We need to explicitly handle the first valid index as a BUY.
        if not weights.empty:
            first_idx = weights.index[0]
            for asset, w in weights.iloc[0].items():
                if abs(w) > 0:
                     # Initial Buy
                     current_val = equity.loc[first_idx] if first_idx in equity.index else self.initial_capital
                     price = prices.at[first_idx, asset] if (first_idx in prices.index and asset in prices.columns) else 0
                     trades.append({
                        'Date': first_idx,
                        'Asset': asset, 
                        'Action': 'BUY',
                        'Weight': round(w, 4),
                        'Value': round(w * current_val, 2),
                        'Price': round(price, 2)
                     })
            
            # Now iterate subsequent changes
            for date, row in diffs.iloc[1:].iterrows():
                # Filter out noise (Vol Targeting Drift)
                # Only log trades if weight change > threshold in total turnover OR specific valid trades
                if row.abs().sum() < threshold: continue
                
                # Equity at the specific date (Rebalance happens at Close)
                curr_equity = equity.loc[date] if date in equity.index else 0
                
                for asset, change in row.items():
                    # Individual asset threshold
                    if abs(change) > threshold:
                        direction = 'BUY' if change > 0 else 'SELL'
                        price = prices.at[date, asset] if (date in prices.index and asset in prices.columns) else 0
                        trades.append({
                            'Date': date,
                            'Asset': asset,
                            'Action': direction,
                            'Weight': round(abs(change), 4),
                            'Value': round(abs(change) * curr_equity, 2),
                            'Price': round(price, 2)
                        })
                        
        return trades[::-1] # Newest first

    def _calculate_metrics(self, returns, drawdown):
        """Calculates generic performance metrics."""
        total_days = len(returns)
        if total_days < 10: 
            return {
                'total_return': 0.0, 'cagr': 0.0, 'sharpe': 0.0, 'sortino': 0.0, 
                'max_drawdown': 0.0, 'win_rate': 0.0, 'calmar': 0.0
            }
        
        # CAGR
        total_ret = (1 + returns).prod() - 1
        ending_value = 1 + total_ret
        years = total_days / 252
        
        if ending_value <= 0:
            cagr = -1.0 # Bust
        elif years > 0:
            cagr = ending_value ** (1/years) - 1
        else:
            cagr = 0.0
        
        # Sharpe
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret > 0:
            sharpe = (mean_ret / std_ret * np.sqrt(252))
        else:
            sharpe = 0.0
        
        # Sortino (Downside Vol)
        downside = returns[returns < 0]
        std_down = downside.std()
        if std_down > 0:
            sortino = (mean_ret / std_down * np.sqrt(252))
        else:
            sortino = 0.0
        
        # Max Drawdown
        max_dd = drawdown.min()
        
        # Win Rate
        if len(returns) > 0:
            win_rate = len(returns[returns > 0]) / len(returns)
        else:
            win_rate = 0.0
            
        # Nan Safety
        calmar = cagr / abs(max_dd) if (max_dd != 0 and not np.isnan(max_dd) and not np.isnan(cagr)) else 0.0

        return {
            'total_return': 0.0 if np.isnan(total_ret) else float(total_ret),
            'cagr': 0.0 if np.isnan(cagr) else float(cagr),
            'sharpe': 0.0 if np.isnan(sharpe) else float(sharpe),
            'sortino': 0.0 if np.isnan(sortino) else float(sortino),
            'max_drawdown': 0.0 if np.isnan(max_dd) else float(max_dd),
            'win_rate': 0.0 if np.isnan(win_rate) else float(win_rate),
            'calmar': 0.0 if np.isnan(calmar) else float(calmar)
        }

    def run_portfolio_simulation(self, prices_df, weights_df, monthly_contribution=0.0, custom_cashflows=None, trade_threshold=0.005):
        """
        Iterative Backtest with Custom Cashflows.
        custom_cashflows: list of dicts [{'Date': 'YYYY-MM-DD', 'Amount': 1000.0}]
        """
        # 1. Align Data
        # 1. Align Data
        asset_returns = self._calculate_safe_returns(prices_df)
        dates = asset_returns.index
        n_days = len(dates)
        
        # 2. Prepare Cashflows Array
        cf_array = np.zeros(n_days)
        
        if custom_cashflows:
            # User provided specific dates
            # Create temp DF to align
            cf_df = pd.DataFrame(custom_cashflows)
            if not cf_df.empty:
                cf_df['Date'] = pd.to_datetime(cf_df['Date'])
                cf_df.set_index('Date', inplace=True)
                # Reindex to simulation dates, sum duplicates, fill 0
                # We need to be careful: if a cashflow is on a non-trading day, roll it forward?
                # Simple approach: reindex with 'nearest' or specific logic?
                # Let's align strictly. If date mismatch (weekend), user should have picked trading day
                # Or we use reindex with tolerance? Better: just match dates.
                
                # Handling weekend inputs: map user dates to nearest trading day GEqual
                for u_date, u_amt in cf_df['Amount'].items():
                     # Find first trading day >= u_date
                     # This is slow if loop.
                     # Faster: searchsorted
                     idx = dates.searchsorted(u_date)
                     if idx < n_days:
                         cf_array[idx] += u_amt
                         
        elif monthly_contribution > 0:
            # Fallback to monthly logic
            # Pre-calculate to avoid logic in loop
            for i in range(1, n_days):
                if dates[i].month != dates[i-1].month:
                    cf_array[i] = monthly_contribution

        # Storage
        equity_series = np.zeros(n_days)
        portfolio_value = self.initial_capital
        
        current_weights = weights_df.iloc[0] 
        if current_weights.sum() > 1.0: current_weights = current_weights / current_weights.sum()
        
        # Equity[0] is initial
        equity_series[0] = portfolio_value
        
        equity_series[0] = portfolio_value
        
        # aligned_weights represents the "Target" we want to be at
        aligned_targets = weights_df.reindex(dates).ffill().fillna(0)
        
        # Current Holdings (as weights of current portfolio value)
        # Init with first day target
        held_weights = aligned_targets.iloc[0].copy()
        
        trades = []
        
        # Init Cost Basis Tracker {Asset: AvgPrice}
        avg_entry_price = {}
        
        # 0. Initial Entry Trades (At t=0)
        # We assume we buy the 'held_weights' at the start date using initial capital
        start_date = dates[0]
        for asset, w in held_weights.items():
            if w > 0:
                trade_val = w * self.initial_capital
                price = prices_df.loc[start_date, asset] if asset in prices_df.columns else 0.0
                avg_entry_price[asset] = price # Initial cost basis
                
                trades.append({
                    'Date': start_date.strftime('%Y-%m-%d'),
                    'Asset': asset,
                    'Action': 'Initial Buy',
                    'Weight': float(w),
                    'Value': float(trade_val),
                    'Price': float(price),
                    'Price': float(price),
                    'PNL_Pct': None,
                    'PNL_Val': None,
                    'Tx_Cost': float(trade_val * self.transaction_cost),
                    'Portfolio_Value': float(portfolio_value)
                })
        
        for i in range(1, n_days):
            date = dates[i]
            
            # 1. Calculate Day's Return on HELD weights
            # The weights we held from yesterday (drifted if no trade)
            r_today = asset_returns.iloc[i]
            day_pnl_pct = (held_weights * r_today).sum()
            
            # Temporary value before costs/cashflows
            # Note: We need to handle value drift to update weights accurately
            # But simplistic pct approach: W_new = W_old * (1+r_asset) / (1+r_port)
            
            # Update Portfolio Value (Gross)
            portfolio_value_gross = portfolio_value * (1 + day_pnl_pct)
            
            # Drift the weights
            denom = (1 + day_pnl_pct)
            if denom != 0:
                held_weights = held_weights * (1 + r_today) / denom
            
            # 2. Check for Rebalancing
            # Target for today (signal generated yesterday close)
            target_weights = aligned_targets.iloc[i]
            
            # Calculate Deviation
            # Total Absolute Deviation / 2 = Turnover required to match
            diff = (target_weights - held_weights).abs().sum() / 2.0
            
            t_cost_pct = 0.0
            
            # Threshold Logic:
            cf_val = cf_array[i]
            
            is_rebalance = False
            if diff > trade_threshold:
                 is_rebalance = True
            elif cf_val != 0:
                 is_rebalance = True
            
            if is_rebalance:
                weight_delta = target_weights - held_weights
                
                # Determine threshold for individual filtering
                asset_threshold = trade_threshold if cf_val == 0 else 0.001
                
                new_held_weights = held_weights.copy()
                executed_turnover = 0.0
                
                for asset, delta in weight_delta.items():
                    if abs(delta) > asset_threshold:
                        action = "Buy" if delta > 0 else "Sell"
                        trade_val = abs(delta) * portfolio_value_gross
                        price = prices_df.loc[date, asset] if asset in prices_df.columns else 0.0
                        
                        trade_rec = {
                            'Date': date.strftime('%Y-%m-%d'),
                            'Asset': asset,
                            'Action': action,
                            'Weight': float(abs(delta)),
                            'Value': float(trade_val),
                            'Price': float(price),
                            'Price': float(price),
                            'PNL_Pct': None,
                            'PNL_Val': None,
                            'Tx_Cost': float(abs(delta) * portfolio_value_gross * self.transaction_cost)
                        }
                        
                        # --- PNL Logic ---
                        current_holding_val = held_weights.get(asset, 0) * portfolio_value_gross
                        current_avg_price = avg_entry_price.get(asset, 0.0)
                        
                        if action == "Sell":
                            # Realized PNL
                            if current_avg_price > 0:
                                # PNL % = (Exit - Entry) / Entry
                                pnl_pct = (price - current_avg_price) / current_avg_price
                                
                                # Approx shares sold
                                shares_sold = trade_val / price if price > 0 else 0
                                pnl_val = (price - current_avg_price) * shares_sold
                                
                                trade_rec['PNL_Pct'] = float(pnl_pct)
                                trade_rec['PNL_Val'] = float(pnl_val)
                                
                                # If sold completely? remove from dict? 
                                # Simpler to keep unless weight -> 0.
                                if target_weights[asset] < 0.0001:
                                     avg_entry_price.pop(asset, None)
                                     
                        elif action == "Buy":
                            # Update Cost Basis (Weighted Average)
                            # Current Shares (estimated)
                            # Avoiding exact share count, estimating via Value / Price
                            old_shares = current_holding_val / price if price > 0 else 0 # Approximation using current price? No.
                            # Better: Shares = Value / CurrentPrice? No.
                            # Shares = Value_Held / Current_Market_Price (at this moment)
                            
                            # Actually, weighted average price formula:
                            # NewAvg = ( (OldShares * OldAvg) + (NewShares * NewPrice) ) / TotalShares
                            # Wait, OldShares * OldAvg = CostBasisValue (Not CurrentMarketValue)
                            
                            # Let's derive shares from current market value
                            curr_shares_est = current_holding_val / price if price > 0 else 0
                            new_shares_est = trade_val / price if price > 0 else 0
                            
                            total_shares = curr_shares_est + new_shares_est
                            
                            # Cost Basis Value (Book Value)
                            book_val_old = curr_shares_est * current_avg_price if current_avg_price > 0 else current_holding_val # fallback
                            book_val_new = new_shares_est * price
                            
                            if total_shares > 0:
                                new_avg = (book_val_old + book_val_new) / total_shares
                                avg_entry_price[asset] = new_avg
                            else:
                                avg_entry_price[asset] = price

                        trades.append(trade_rec)
                        
                        # We traded this asset -> Update Held to Target
                        new_held_weights[asset] = target_weights[asset]
                        executed_turnover += abs(delta)
                    else:
                        # Did NOT trade -> Held weight remains drifted
                        pass

                # Cost on executed volume only
                t_cost_pct = executed_turnover * self.transaction_cost
                
                # Update Holdings
                held_weights = new_held_weights
            
            # 3. Apply Cost and Net Value
            net_ret = day_pnl_pct - t_cost_pct
            
            # Apply Return
            portfolio_value = portfolio_value * (1 + net_ret)

            # Update trades with final daily equity
            for t in trades:
                if t['Date'] == date.strftime('%Y-%m-%d') and 'Portfolio_Value' not in t:
                    t['Portfolio_Value'] = float(portfolio_value)

            
            # 4. Add Cashflow (End of Day)
            if cf_val != 0:
                # Generate trades showing how contribution is allocated
                for asset, target_wt in target_weights.items():
                    if target_wt > 0.001:  # Only for assets we're buying
                        contribution_allocation = abs(cf_val) * target_wt
                        price = prices_df.loc[date, asset] if asset in prices_df.columns else 0.0
                        
                        trades.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Asset': asset,
                            'Action': 'BUY' if cf_val > 0 else 'SELL',
                            'Weight': float(target_wt),
                            'Value': float(contribution_allocation),
                            'Price': float(price),
                            'PNL_Pct': None,
                            'PNL_Val': None,
                            'Tx_Cost': float(contribution_allocation * self.transaction_cost),
                            'Portfolio_Value': float(portfolio_value + cf_val)
                        })
                        
                        # Update cost basis for new buys
                        if cf_val > 0:
                            curr_val = held_weights.get(asset, 0) * portfolio_value
                            curr_shares = curr_val / price if price > 0 else 0
                            new_shares = contribution_allocation / price if price > 0 else 0
                            curr_avg = avg_entry_price.get(asset, price)
                            
                            total_shares = curr_shares + new_shares
                            if total_shares > 0:
                                new_avg = ((curr_shares * curr_avg) + (new_shares * price)) / total_shares
                                avg_entry_price[asset] = new_avg
                
                portfolio_value += cf_val
                # Cashflow enters, but we already set weights to target?
                # If we add cash, strictly speaking, it's cash until invested.
                # But our logic assumes "Target Weights" define full exposure.
                # So if we added cash, we assume it was bought into Target allocation immediately (at Close).
            
            equity_series[i] = portfolio_value
        
        equity_curve = pd.Series(equity_series, index=dates)
        
        # Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        return {
            'equity_curve': equity_curve,
            'drawdown': drawdown,
            'metrics': self._calculate_metrics(equity_curve.pct_change().fillna(0), drawdown),
            'trades': trades,
            'cash_flows': pd.Series(cf_array, index=dates),
            'latest_status': {
                'date': dates[-1].strftime('%Y-%m-%d'),
                'total_equity': float(portfolio_value),
                'held_weights': held_weights.to_dict(),
                'target_weights': aligned_targets.iloc[-1].to_dict()
            }
        }
