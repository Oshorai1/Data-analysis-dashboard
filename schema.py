from typing import Dict, Any, List
import pandas as pd
import numpy as np
import re

ts_name_re = re.compile(r"(^|_)(ts|time|timestamp|date|datetime|dt)($|_)", flags=re.I)
id_name_re = re.compile(r"(^|_)(id|order_id|trade_id|tx_id|uid|user_id)($|_)", flags=re.I)

def infer_schema(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cols = list(df.columns)
    types = {"numeric": [], "categorical": [], "datetime": [], "boolean": [], "text": [], "id": []}
    mapping: Dict[str, List[str]] = {}

    for c in cols:
        s = df[c]
        sval = str(c).lower()
        vals = s.dropna()

        # ID by name or high-uniqueness
        if id_name_re.search(sval) or (len(vals) > 0 and vals.nunique() > 0.9 * max(1, len(vals))):
            types['id'].append(c)
            mapping[c] = ['id']
            continue

        # boolean
        try:
            uniq = set(list(vals.unique()[:20]))
            if uniq and all(str(u).lower() in ('0','1','true','false') for u in uniq):
                types['boolean'].append(c)
                mapping[c] = ['boolean']
                continue
        except Exception:
            pass

        # datetime by name
        if ts_name_re.search(sval):
            parsed = pd.to_datetime(s, errors='coerce', utc=True)
            if parsed.notna().any():
                types['datetime'].append(c)
                mapping[c] = ['datetime']
                continue

        # numeric detection
        if pd.api.types.is_numeric_dtype(s):
            # timestamp heuristics by scale
            if not vals.empty:
                vmin = vals.min(); vmax = vals.max()
                try:
                    if vmin > 1e9 and vmax < 1e11:
                        types['datetime'].append(c); mapping[c]=['datetime','unix_seconds']; continue
                    if vmin > 1e12 and vmax < 1e15:
                        types['datetime'].append(c); mapping[c]=['datetime','unix_milliseconds']; continue
                    if vmin > 1e15 and vmax < 1e18:
                        types['datetime'].append(c); mapping[c]=['datetime','unix_microseconds']; continue
                    if vmin > 1e18:
                        types['datetime'].append(c); mapping[c]=['datetime','unix_nanoseconds']; continue
                except Exception:
                    pass

            # low-cardinality numeric -> categorical
            try:
                nunique = int(s.nunique(dropna=True))
            except Exception:
                nunique = 0
            if nunique > 0 and nunique < 0.05 * max(1, len(s)) and nunique < cfg.get('max_onehot_categories', 50):
                types['categorical'].append(c)
                mapping[c] = ['categorical']
            else:
                types['numeric'].append(c)
                mapping[c] = ['numeric']
            continue

        # string-like
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            sample = vals.astype(str).head(200)
            avg_len = sample.map(len).mean() if not sample.empty else 0
            uniq_ratio = sample.nunique() / max(1, len(sample)) if not sample.empty else 0
            if avg_len > 50 or uniq_ratio > 0.7:
                types['text'].append(c); mapping[c]=['text']
            else:
                types['categorical'].append(c); mapping[c]=['categorical']
            continue

        # fallback
        types['categorical'].append(c); mapping[c]=['categorical']

    return {"types": types, "mapping": mapping}


def infer_column_roles(df: pd.DataFrame) -> Dict[str, str]:
    """Return mapping of semantic role -> column name where detected."""
    roles = {}
    cols = list(df.columns)
    lowered = {c: str(c).lower() for c in cols}

    def find(*keys):
        for k in keys:
            for c, lc in lowered.items():
                if k in lc:
                    return c
        return None

    # trading roles
    roles['datetime'] = find('datetime','timestamp','time','ts','date')
    roles['symbol'] = find('symbol','ticker','instrument')
    roles['open'] = find('open')
    roles['high'] = find('high')
    roles['low'] = find('low')
    roles['close'] = find('close','ltp','last_price','price')
    roles['volume'] = find('volume','vol')
    roles['qty'] = find('qty','quantity','size')
    roles['side'] = find('side','direction')
    roles['entry_price'] = find('entry','entry_price')
    roles['exit_price'] = find('exit','exit_price')
    roles['realized_pnl'] = find('realized_pnl','realized_pnl_month','realized','pnl','pl')
    roles['unrealized_pnl'] = find('unrealized','unrealized_pnl')
    roles['equity'] = find('equity','balance','nav')
    roles['fees'] = find('fee','fees','commission')
    roles['drawdown_pct'] = find('drawdown','max_drawdown')

    # clinic roles
    roles['patient_id'] = find('patient_id','patient','pid')
    roles['visit_id'] = find('visit_id','visit')
    roles['test_name'] = find('test_name','lab','test')
    roles['test_value'] = find('value','result','test_value')
    roles['unit'] = find('unit')
    roles['abnormal_flag'] = find('abnormal','flag')
    roles['doctor'] = find('doctor','physician','provider')
    roles['department'] = find('department','dept')
    roles['billing_amount'] = find('billed','billed_amount','amount','charges','fee','total','paid','paid_amount','paid_amt')
    
    # backtest roles
    roles['month'] = find('month','period','month_start','month_end')
    roles['trades'] = find('trades','total_trades','trades_count')
    roles['end_equity'] = find('end_equity','equity_end','final_equity')

    # clean up None values
    roles = {k: v for k, v in roles.items() if v}
    return roles


def infer_dataset_type(df: pd.DataFrame, roles: Dict[str,str]) -> str:
    """Detect dataset type: trading vs clinic vs generic.
    
    Trading types: trade_log, order_log, equity_curve, backtest_monthly_summary
    Clinic types: lab_results, vitals, appointments, billing
    """
    cols_lower = [str(c).lower() for c in df.columns]
    cols_str = ' '.join(cols_lower)
    nrows = len(df)
    
    # TRADING: backtest_monthly_summary (small file with monthly telemetry)
    # Indicators: has month col + 4+ backtest fields
    has_month = roles.get('month') is not None or any('month' in c for c in cols_lower)
    backtest_fields = [
        'trades' in cols_str,
        'realized_pnl' in cols_str,
        'end_equity' in cols_str,
        'max_drawdown' in cols_str,
        'total_fees' in cols_str,
        'avg_epsilon' in cols_str,
        'agent_train_steps' in cols_str,
        'blocked_total' in cols_str,
    ]
    if has_month and sum(backtest_fields) >= 4:
        return 'backtest_monthly_summary'
    
    # TRADING: order_log (order-centric)
    # Indicators: has order_id + status + order_type + quantity
    has_order_id = any(k in cols_str for k in ['order_id', 'orderid', 'order_no'])
    has_status = any(k in cols_str for k in ['status', 'order_status', 'state'])
    has_order_type = any(k in cols_str for k in ['order_type', 'type', 'side'])
    if has_order_id and has_status and has_order_type and nrows > 10:
        return 'order_log'
    
    # TRADING: OHLCV candle data
    has_ohlcv = roles.get('open') and roles.get('high') and roles.get('low') and roles.get('close')
    if has_ohlcv and roles.get('volume') and nrows > 10:
        return 'ohlcv_candles'
    
    # TRADING: trade_log (trade-centric)
    # Indicators: (entry/exit price OR filled price) + qty + symbol + pnl
    has_entry_exit = roles.get('entry_price') or roles.get('exit_price')
    has_qty = any(k in cols_str for k in ['qty', 'quantity', 'size'])
    has_symbol = roles.get('symbol') is not None
    has_pnl = roles.get('realized_pnl') or roles.get('pnl')
    if nrows > 10 and has_qty and (has_entry_exit or has_pnl or has_symbol):
        return 'trade_log'
    
    # TRADING: equity_curve (time-series equity)
    has_equity = roles.get('equity') and roles.get('datetime')
    if has_equity and not has_qty and nrows > 50:
        return 'equity_curve'

    # CLINIC types
    if roles.get('test_name') and roles.get('test_value'):
        return 'lab_results_table'
    if any(k in cols_str for k in ('bp','blood_pressure','systolic','diastolic','hr','heart_rate','spo2')):
        return 'vitals_table'
    if roles.get('visit_id') or roles.get('doctor'):
        return 'appointments_or_visits_table'
    if roles.get('billing_amount'):
        return 'billing_or_revenue_table'

    return 'generic_table'
