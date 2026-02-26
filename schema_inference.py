"""Schema inference helpers.
Small, dependency-free helpers to infer numeric/categorical/datetime/text/id columns.
These are a convenience wrapper for use by the engine and UI.
"""
from typing import Dict, List
import pandas as pd
import re


def infer_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    out = {
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "id_columns": [],
        "boolean_columns": [],
    }

    id_name_re = re.compile(r"(^|_)(id|order_id|trade_id|tx_id|uid|user_id)($|_)", flags=re.I)
    dt_name_re = re.compile(r"(^|_)(ts|time|timestamp|date|datetime|dt|created_at)($|_)", flags=re.I)

    for c in df.columns:
        series = df[c]
        sval = c.lower()
        vals_nonnull = series.dropna()

        if id_name_re.search(sval):
            out["id_columns"].append(c)
            continue

        # booleans
        unique_vals = set(list(vals_nonnull.unique()[:20])) if len(vals_nonnull) > 0 else set()
        if unique_vals and all((v in (0, 1, True, False) or str(v).lower() in ("0", "1", "true", "false")) for v in unique_vals):
            out["boolean_columns"].append(c)
            out["categorical_columns"].append(c)
            continue

        # datetime by name
        if dt_name_re.search(sval):
            try:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed is not None and parsed.notna().sum() > 0:
                    out["datetime_columns"].append(c)
                    continue
            except Exception:
                pass

        # numeric
        if pd.api.types.is_numeric_dtype(series):
            if not series.dropna().empty:
                vmin = series.min()
                vmax = series.max()
                # seconds
                try:
                    if vmin > 1e9 and vmax < 1e11:
                        p = pd.to_datetime(series, unit="s", errors="coerce")
                        if p.notna().sum() > 0:
                            out["datetime_columns"].append(c)
                            continue
                    if vmin > 1e12 and vmax < 1e15:
                        p = pd.to_datetime(series, unit="ms", errors="coerce")
                        if p.notna().sum() > 0:
                            out["datetime_columns"].append(c)
                            continue
                except Exception:
                    pass

            try:
                nunique = int(series.nunique(dropna=True))
            except Exception:
                nunique = 0
            if nunique > 0 and nunique < 0.05 * max(1, len(series)) and nunique < 50:
                out["categorical_columns"].append(c)
            else:
                out["numeric_columns"].append(c)
            continue

        # text
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            sample = vals_nonnull.astype(str).head(200)
            avg_len = sample.map(len).mean() if not sample.empty else 0
            unique_ratio = sample.nunique() / max(1, len(sample)) if not sample.empty else 0
            if avg_len > 50 or unique_ratio > 0.7:
                out["text_columns"].append(c)
            else:
                out["categorical_columns"].append(c)
            continue

        out["categorical_columns"].append(c)

    return out
