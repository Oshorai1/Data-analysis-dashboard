from typing import Tuple, List
from pathlib import Path
import pandas as pd
from .errors import ValidationError

def validate_table(df: pd.DataFrame, path: str, cfg: dict) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warnings = []
    errors = []
    p = Path(path)
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > cfg.get('max_upload_mb', 50):
        errors.append(f"File size {size_mb:.1f}MB exceeds max {cfg.get('max_upload_mb')}MB")

    if df is None or df.empty:
        errors.append('Empty or unreadable table')

    if df.shape[0] < 10:
        warnings.append('Very small number of rows (<10)')

    # duplicate columns
    cols = list(df.columns)
    dup = [c for c in set(cols) if cols.count(c) > 1]
    if dup:
        warnings.append(f'Duplicate columns detected: {dup}')

    # too many missing columns
    na_frac = df.isna().mean()
    many_missing = na_frac[na_frac > 0.95].index.tolist()
    if many_missing:
        warnings.append(f'Columns with >95% missing values: {many_missing[:10]}')

    # malformed CSV detection: single column with many commas -> warn
    if df.shape[1] == 1:
        s = df.iloc[:,0].astype(str).head(100).tolist()
        comma_lines = sum(1 for r in s if ',' in r)
        if comma_lines > 5:
            warnings.append('File might be malformed CSV (single-column parse).')

    return df, warnings, errors
