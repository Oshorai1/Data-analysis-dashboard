from typing import Dict, Any
import pandas as pd

def profile(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out['shape'] = {'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
    mem = df.memory_usage(deep=True).sum()
    out['memory_bytes'] = int(mem)
    # missingness
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    out['missing_top'] = miss.reset_index().rename(columns={'index':'column',0:'missing'}) .to_dict('records') if not miss.empty else []
    # numeric describe
    num = df.select_dtypes(include=['number'])
    if not num.empty:
        out['numeric_describe'] = num.describe().T.reset_index().rename(columns={'index':'column'}).to_dict('records')
    # categorical top values
    cats = []
    for c in df.select_dtypes(include=['object','category']).columns[:20]:
        vc = df[c].astype(str).value_counts(dropna=False).head(10)
        cats.append({'column': c, 'top_values': vc.to_dict()})
    out['categorical_top_values'] = cats
    # datetime ranges
    dts = []
    for c in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
        ser = df[c].dropna()
        if not ser.empty:
            dts.append({'column': c, 'min': str(ser.min()), 'max': str(ser.max()), 'n': int(len(ser))})
    out['datetime_summary'] = dts
    out['duplicate_rows'] = int(df.duplicated().sum())
    return out
