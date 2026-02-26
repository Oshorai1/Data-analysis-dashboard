from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import re

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    old = list(df.columns)
    new = []
    for c in old:
        s = str(c).strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "", s)
        if not s:
            s = "col"
        new.append(s)
    df = df.copy()
    df.columns = new
    return df, [f"{o} -> {n}" for o, n in zip(old, new)]

def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    nun = df.nunique(dropna=False)
    const = [c for c, v in nun.items() if v <= 1]
    if const:
        df = df.drop(columns=const, errors='ignore')
    return df, const

def handle_missing(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    log = []
    # numeric median
    for c in df.select_dtypes(include=['number']).columns:
        if df[c].isna().mean() > cfg.get('missing_threshold_drop', 0.5):
            continue
        med = df[c].median()
        df[c] = df[c].fillna(med)
    # categorical fill
    for c in df.select_dtypes(include=['object', 'category']).columns:
        df[c] = df[c].fillna('missing')
    # infinities
    for c in df.select_dtypes(include=['number']).columns:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())
    return df, log

def build_model_frame(df: pd.DataFrame, schema: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    types = schema.get('types', {})
    numeric = types.get('numeric', [])
    ids = set(types.get('id', []))
    datetimes = set(types.get('datetime', []))
    text_cols = set(types.get('text', []))

    # exclude columns
    cols = [c for c in numeric if c not in ids and c not in datetimes and c not in text_cols]
    # drop columns with >50% missing
    cols = [c for c in cols if df[c].notna().mean() >= (1 - cfg.get('missing_threshold_drop', 0.5))]
    # near-zero variance
    keep = []
    for c in cols:
        try:
            if df[c].nunique(dropna=True) > 1:
                keep.append(c)
        except Exception:
            continue
    max_cols = cfg.get('max_model_columns', 200)
    keep = keep[:max_cols]
    return df[keep].copy()
