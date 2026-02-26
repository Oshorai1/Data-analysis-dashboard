"""Trading KPI Engine - Handles all trading dataset types."""
from .base import KPIEngineBase, KPIResult
from .utils import resolve_mapping, safe_float, safe_parse_json
from .formatting import currency_fmt
import pandas as pd
import numpy as np
import json
import math
from collections import defaultdict


class TradingKPIEngine(KPIEngineBase):
    mode_name = 'trading'

    def compute(self, pr, config=None):
        cfg = config or self.cfg
        res = KPIResult(mode='trading')
        df = pr.df_clean
        mapping = resolve_mapping(pr.mapping or {}, pr.column_profile or {})

        if df is None or len(df) == 0:
            res.warnings.append('No table data for trading KPIs')
            res.summary = 'No table data to analyze.'
            return res

        def col(k):
            return mapping.get(k)

        # Use pipeline's dataset type detection if available
        dataset_type = (pr.column_profile.get('dataset_type')
                       if isinstance(pr.column_profile, dict) else None) or 'generic'

        # Route to appropriate handler
        if dataset_type == 'backtest_monthly_summary':
            return self._handle_backtest_monthly_summary(df, mapping, res, cfg)
        elif dataset_type == 'trade_log':
            return self._handle_trade_log(df, mapping, res, cfg)
        elif dataset_type == 'equity_curve':
            return self._handle_equity_curve(df, mapping, res, cfg)
        elif dataset_type == 'order_log':
            return self._handle_order_log(df, mapping, res, cfg)
        elif dataset_type == 'ohlcv_candles':
            return self._handle_ohlcv(df, mapping, res, cfg)
        else:
            # Generic fallback
            return self._handle_generic(df, mapping, res, cfg)

    def _handle_backtest_monthly_summary(self, df, mapping, res, cfg):
        """Handle monthly backtest summary datasets (9-12 rows)."""
        try:
            def col(k):
                return mapping.get(k)

            # Find month column
            month_col = col('month') or next((c for c in df.columns if 'month' in str(c).lower()), None)
            if month_col is None or month_col not in df.columns:
                return res

            months = df[month_col].astype(str).tolist()

            def num(col_name):
                c = col(col_name)
                if c and c in df.columns:
                    return pd.to_numeric(df[c], errors='coerce')
                return pd.Series([np.nan] * len(df))

            # Extract key columns
            realized_pnl = num('realized_pnl')
            end_equity = num('end_equity')
            max_dd = num('drawdown_pct')
            trades = num('trades')
            fees = num('fees')
            steps = num('agent_train_steps')
            epsilon = num('avg_epsilon')
            blocked = num('blocked_total')

            # --- COMPUTED METRICS ---
            res.computed['total_months'] = int(len(df))
            res.computed['total_realized_pnl'] = float(realized_pnl.sum(skipna=True)) if realized_pnl.notna().any() else 0
            res.computed['avg_realized_pnl_per_month'] = float(realized_pnl.mean(skipna=True)) if realized_pnl.notna().any() else 0
            res.computed['std_realized_pnl'] = float(realized_pnl.std(skipna=True)) if realized_pnl.notna().any() else 0

            # Best/worst months
            if realized_pnl.notna().any():
                idx_best = int(realized_pnl.idxmax())
                idx_worst = int(realized_pnl.idxmin())
                res.computed['best_month_realized_pnl'] = months[idx_best]
                res.computed['best_month_realized_pnl_value'] = float(realized_pnl.iloc[idx_best])
                res.computed['worst_month_realized_pnl'] = months[idx_worst]
                res.computed['worst_month_realized_pnl_value'] = float(realized_pnl.iloc[idx_worst])

            if end_equity.notna().any():
                idx_best_eq = int(end_equity.idxmax())
                idx_worst_eq = int(end_equity.idxmin())
                res.computed['best_month_end_equity'] = months[idx_best_eq]
                res.computed['best_month_end_equity_value'] = float(end_equity.iloc[idx_best_eq])
                res.computed['worst_month_end_equity'] = months[idx_worst_eq]

            if max_dd.notna().any():
                idx_worst_dd = int(max_dd.idxmin())  # Most negative drawdown
                res.computed['worst_drawdown_month'] = months[idx_worst_dd]
                res.computed['worst_drawdown_pct'] = float(max_dd.iloc[idx_worst_dd])
                res.computed['max_drawdown_pct'] = float(max_dd.min())

            # Trade statistics
            if trades.notna().any():
                res.computed['total_trades'] = int(trades.sum(skipna=True))
                res.computed['avg_trades_per_month'] = float(trades.mean(skipna=True))
                res.computed['max_trades_per_month'] = int(trades.max())
                res.computed['min_trades_per_month'] = int(trades.min())

            # Fee statistics
            if fees.notna().any():
                res.computed['total_fees_paid'] = float(fees.sum(skipna=True))
                res.computed['avg_fees_per_month'] = float(fees.mean(skipna=True))
                res.computed['fees_as_pct_of_pnl'] = (
                    float(fees.sum() / abs(realized_pnl.sum() + 0.001) * 100)
                    if realized_pnl.sum() != 0 else 0
                )

            # Training telemetry
            if steps.notna().any():
                res.computed['total_training_steps'] = int(steps.sum(skipna=True))
                res.computed['avg_training_steps_per_month'] = float(steps.mean(skipna=True))

            if epsilon.notna().any():
                res.computed['avg_epsilon'] = float(epsilon.mean(skipna=True))
                res.computed['epsilon_trend'] = (
                    'decreasing (exploration reduced)' if epsilon.iloc[-1] < epsilon.iloc[0] else
                    'increasing (exploration increased)' if epsilon.iloc[-1] > epsilon.iloc[0] else
                    'stable'
                )

            # Blocked trades
            if blocked.notna().any():
                res.computed['total_blocked_trades'] = int(blocked.sum(skipna=True))
                res.computed['avg_blocked_per_month'] = float(blocked.mean(skipna=True))

            # PnL to drawdown ratio (risk efficiency)
            if (max_dd.notna().any() and realized_pnl.notna().any() and max_dd.min() < 0):
                res.computed['pnl_to_max_drawdown_ratio'] = (
                    float(abs(realized_pnl.sum()) / abs(max_dd.min()))
                    if max_dd.min() != 0 else None
                )

            # --- TABLES ---
            # Monthly time series
            rows = []
            for i, m in enumerate(months):
                rows.append({
                    'month': m,
                    'realized_pnl': float(realized_pnl.iloc[i]) if not pd.isna(realized_pnl.iloc[i]) else None,
                    'end_equity': float(end_equity.iloc[i]) if not pd.isna(end_equity.iloc[i]) else None,
                    'max_drawdown_pct': float(max_dd.iloc[i]) if not pd.isna(max_dd.iloc[i]) else None,
                    'trades': int(trades.iloc[i]) if not pd.isna(trades.iloc[i]) else None,
                })
            res.tables['monthly_summary'] = rows

            # --- CHARTS ---
            res.charts.append({
                'id': 'backtest_realized_pnl_by_month',
                'chart_type': 'bar',
                'title': 'Monthly Realized PnL',
                'data': [{'month': m, 'realized_pnl': p} for m, p in zip(months, realized_pnl.fillna(0).tolist())],
                'priority': 1
            })
            res.charts.append({
                'id': 'backtest_end_equity_by_month',
                'chart_type': 'line',
                'title': 'End Equity Trajectory',
                'data': [{'month': m, 'end_equity': e} for m, e in zip(months, end_equity.fillna(0).tolist())],
                'priority': 1
            })
            res.charts.append({
                'id': 'backtest_drawdown_by_month',
                'chart_type': 'bar',
                'title': 'Max Drawdown % by Month',
                'data': [{'month': m, 'drawdown_pct': d} for m, d in zip(months, max_dd.fillna(0).tolist())],
                'priority': 2
            })
            res.charts.append({
                'id': 'backtest_trades_by_month',
                'chart_type': 'bar',
                'title': 'Trades Executed per Month',
                'data': [{'month': m, 'trades': t} for m, t in zip(months, trades.fillna(0).tolist())],
                'priority': 2
            })
            if steps.notna().any():
                res.charts.append({
                    'id': 'backtest_training_steps',
                    'chart_type': 'line',
                    'title': 'Agent Training Steps by Month',
                    'data': [{'month': m, 'steps': s} for m, s in zip(months, steps.fillna(0).tolist())],
                    'priority': 3
                })
            if epsilon.notna().any():
                res.charts.append({
                    'id': 'backtest_epsilon_trend',
                    'chart_type': 'line',
                    'title': 'Avg Epsilon (Exploration Rate) by Month',
                    'data': [{'month': m, 'epsilon': e} for m, e in zip(months, epsilon.fillna(0).tolist())],
                    'priority': 3
                })

            # --- NARRATIVE ---
            parts = []
            parts.append(f"This is a backtest monthly summary covering {res.computed['total_months']} months.")
            if res.computed.get('total_realized_pnl') is not None:
                total_pnl = res.computed['total_realized_pnl']
                parts.append(f"Total realized PnL: {currency_fmt(total_pnl, cfg.currency)}.")
                if res.computed.get('avg_realized_pnl_per_month') is not None:
                    parts.append(f"Average monthly PnL: {currency_fmt(res.computed['avg_realized_pnl_per_month'], cfg.currency)}.")
            if res.computed.get('best_month_realized_pnl') is not None:
                best_val = res.computed.get('best_month_realized_pnl_value', 0)
                parts.append(f"Best month: {res.computed['best_month_realized_pnl']} ({currency_fmt(best_val, cfg.currency)}).")
            if res.computed.get('worst_month_realized_pnl') is not None:
                worst_val = res.computed.get('worst_month_realized_pnl_value', 0)
                parts.append(f"Worst month: {res.computed['worst_month_realized_pnl']} ({currency_fmt(worst_val, cfg.currency)}).")
            if res.computed.get('worst_drawdown_pct') is not None:
                dd = res.computed['worst_drawdown_pct']
                dd_desc = 'severe' if dd < -0.20 else 'significant' if dd < -0.10 else 'moderate'
                parts.append(f"Worst drawdown: {dd:.2%} in {res.computed.get('worst_drawdown_month', '?')} ({dd_desc}).")
            if res.computed.get('total_fees_paid') is not None:
                parts.append(f"Total fees paid: {currency_fmt(res.computed['total_fees_paid'], cfg.currency)}.")
            if res.computed.get('epsilon_trend') is not None:
                parts.append(f"Epsilon trend: {res.computed['epsilon_trend']}.")
            if res.computed.get('total_blocked_trades') is not None and res.computed['total_blocked_trades'] > 0:
                parts.append(f"Total blocked trades: {res.computed['total_blocked_trades']}.")

            res.summary = ' '.join(parts)
            res.used_columns = {'month': month_col or '?', 'realized_pnl': col('realized_pnl') or '?'}
            return res

        except Exception as e:
            res.warnings.append(f'Backtest summary processing failed: {e}')
            return res

    def _handle_equity_curve(self, df, mapping, res, cfg):
        """Handle equity curve datasets (time-series equity with dates)."""
        try:
            def col(k):
                return mapping.get(k)

            dt_col = col('datetime')
            eq_col = col('equity')

            if not all([dt_col, dt_col in df.columns, eq_col, eq_col in df.columns]):
                res.warnings.append('Missing datetime or equity column for equity curve analysis')
                return res

            dates = pd.to_datetime(df[dt_col], errors='coerce')
            equity = pd.to_numeric(df[eq_col], errors='coerce').fillna(method='ffill').fillna(0)

            # --- COMPUTED METRICS ---
            if len(equity) > 0:
                start_eq = float(equity.iloc[0])
                end_eq = float(equity.iloc[-1])
                res.computed['start_equity'] = start_eq
                res.computed['end_equity'] = end_eq
                res.computed['total_return_amount'] = end_eq - start_eq
                if start_eq != 0:
                    res.computed['total_return_pct'] = ((end_eq - start_eq) / abs(start_eq)) * 100

                # Drawdown
                peak = equity.cummax()
                drawdown = equity - peak
                res.computed['max_drawdown_amount'] = float(drawdown.min())
                if start_eq != 0:
                    res.computed['max_drawdown_pct'] = (float(drawdown.min()) / abs(start_eq)) * 100

                # Volatility
                returns = equity.pct_change().dropna()
                if len(returns) > 1:
                    res.computed['daily_return_volatility'] = float(returns.std() * 100) if not returns.empty else None
                    res.computed['annualized_volatility'] = float(returns.std() * np.sqrt(252) * 100) if not returns.empty else None

                # Sharpe-like ratio
                if len(returns) > 30 and returns.std() > 0:
                    res.computed['sharpe_ratio_proxy'] = float((returns.mean() / returns.std()) * np.sqrt(252))

                # Longest drawdown duration
                try:
                    below_peak = (equity < peak).astype(int)
                    dd_duration = (below_peak.diff() != 0).cumsum()
                    res.computed['longest_drawdown_duration_days'] = int(
                        (below_peak == 1).groupby((below_peak != below_peak.shift()).cumsum()).sum().max()
                    )
                except Exception:
                    pass

            # --- CHARTS ---
            res.charts.append({
                'id': 'equity_curve_ts',
                'chart_type': 'equity_curve',
                'title': 'Equity Curve Over Time',
                'x': dt_col,
                'y': eq_col,
                'priority': 1
            })
            res.charts.append({
                'id': 'drawdown_curve',
                'chart_type': 'line',
                'title': 'Drawdown Curve',
                'x': dt_col,
                'y': 'drawdown',
                'priority': 1
            })

            # --- NARRATIVE ---
            parts = []
            parts.append(f"Equity curve analysis with {len(df)} data points.")
            if res.computed.get('total_return_pct') is not None:
                parts.append(f"Total return: {res.computed['total_return_pct']:.2f}%.")
            if res.computed.get('max_drawdown_pct') is not None:
                parts.append(f"Maximum drawdown: {res.computed['max_drawdown_pct']:.2f}%.")
            if res.computed.get('sharpe_ratio_proxy') is not None:
                parts.append(f"Sharpe ratio (proxy): {res.computed['sharpe_ratio_proxy']:.2f}.")
            res.summary = ' '.join(parts)
            res.used_columns = {'datetime': dt_col or '?', 'equity': eq_col or '?'}
            return res

        except Exception as e:
            res.warnings.append(f'Equity curve processing failed: {e}')
            return res

    def _handle_trade_log(self, df, mapping, res, cfg):
        """Handle trade log datasets (individual trade records)."""
        try:
            def col(k):
                return mapping.get(k)

            pnl_col = col('realized_pnl') or col('pnl')
            symbol_col = col('symbol')
            side_col = col('side')
            fees_col = col('fees')
            dt_col = col('datetime')

            # Basic trade count
            res.computed['total_trades'] = int(len(df))

            # PnL statistics
            if pnl_col and pnl_col in df.columns:
                pnl = pd.to_numeric(df[pnl_col], errors='coerce').fillna(0)
                total_pnl = float(pnl.sum())
                res.computed['total_pnl'] = total_pnl
                wins = pnl[pnl > 0]
                losses = pnl[pnl <= 0]
                res.computed['win_rate'] = float(len(wins) / max(1, len(pnl)))
                res.computed['avg_win'] = float(wins.mean()) if len(wins) > 0 else 0
                res.computed['avg_loss'] = float(losses.mean()) if len(losses) > 0 else 0
                gross_win = float(wins.sum())
                gross_loss = abs(float(losses.sum()))
                res.computed['profit_factor'] = (gross_win / gross_loss) if gross_loss > 0 else None

                # PnL distribution chart
                res.charts.append({
                    'id': 'trade_pnl_histogram',
                    'chart_type': 'hist',
                    'title': 'Trade PnL Distribution',
                    'y': pnl_col,
                    'priority': 2
                })

            # Symbol breakdown
            if symbol_col and symbol_col in df.columns and pnl_col and pnl_col in df.columns:
                try:
                    by_symbol = df.groupby(symbol_col)[pnl_col].apply(
                        lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()
                    ).sort_values(ascending=False).head(10)
                    res.tables['pnl_by_symbol'] = [{'symbol': str(k), 'pnl': float(v)} for k, v in by_symbol.items()]
                    res.charts.append({
                        'id': 'trades_pnl_by_symbol',
                        'chart_type': 'bar',
                        'title': 'PnL by Symbol (Top 10)',
                        'data': res.tables['pnl_by_symbol'],
                        'priority': 2
                    })
                except Exception:
                    pass

            # Side breakdown (long vs short)
            if side_col and side_col in df.columns and pnl_col and pnl_col in df.columns:
                try:
                    by_side = df.groupby(side_col)[pnl_col].apply(
                        lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()
                    )
                    res.tables['pnl_by_side'] = [{'side': str(k), 'pnl': float(v)} for k, v in by_side.items()]
                except Exception:
                    pass

            # Cumulative PnL over time
            if dt_col and dt_col in df.columns and pnl_col and pnl_col in df.columns:
                try:
                    df_sorted = df.sort_values(dt_col).copy()
                    cumulative_pnl = pd.to_numeric(df_sorted[pnl_col], errors='coerce').fillna(0).cumsum()
                    res.charts.append({
                        'id': 'trades_cumulative_pnl',
                        'chart_type': 'line',
                        'title': 'Cumulative PnL Over Time',
                        'x': dt_col,
                        'y': pnl_col,
                        'priority': 1
                    })
                except Exception:
                    pass

            # --- NARRATIVE ---
            parts = []
            parts.append(f"Trade log with {res.computed['total_trades']} trades.")
            if res.computed.get('total_pnl') is not None:
                parts.append(f"Total PnL: {currency_fmt(res.computed['total_pnl'], cfg.currency)}.")
            if res.computed.get('win_rate') is not None:
                parts.append(f"Win rate: {res.computed['win_rate']*100:.1f}%.")
            if res.computed.get('profit_factor') is not None:
                parts.append(f"Profit factor: {res.computed['profit_factor']:.2f}.")
            res.summary = ' '.join(parts)
            res.used_columns = {'pnl': pnl_col or '?', 'symbol': symbol_col or '?'}
            return res

        except Exception as e:
            res.warnings.append(f'Trade log processing failed: {e}')
            return res

    def _handle_order_log(self, df, mapping, res, cfg):
        """Handle order log datasets (order-centric)."""
        try:
            def col(k):
                return mapping.get(k)

            res.computed['total_orders'] = int(len(df))

            # Fill rate
            status_col = next((c for c in df.columns if 'status' in str(c).lower()), None)
            if status_col:
                statuses = df[status_col].astype(str).str.lower()
                filled = int((statuses.str.contains('filled|complete')).sum())
                res.computed['fill_rate'] = float(filled / len(df)) if len(df) > 0 else 0
                rejected = int((statuses.str.contains('reject|cancel')).sum())
                res.computed['rejection_rate'] = float(rejected / len(df)) if len(df) > 0 else 0

            res.summary = f"Order log with {res.computed['total_orders']} orders."
            return res

        except Exception as e:
            res.warnings.append(f'Order log processing failed: {e}')
            return res

    def _handle_ohlcv(self, df, mapping, res, cfg):
        """Handle OHLCV candle data."""
        try:
            def col(k):
                return mapping.get(k)

            res.computed['total_bars'] = int(len(df))
            res.summary = f"OHLCV candle data with {len(df)} bars."
            return res

        except Exception as e:
            res.warnings.append(f'OHLCV processing failed: {e}')
            return res

    def _handle_generic(self, df, mapping, res, cfg):
        """Fallback generic handler."""
        res.computed['row_count'] = int(len(df))
        res.computed['column_count'] = int(df.shape[1])
        res.summary = f"Generic trading data: {len(df)} rows, {df.shape[1]} columns."
        return res
