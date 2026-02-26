from .base import KPIEngineBase, KPIResult
import pandas as pd
import numpy as np

class UniversalKPIEngine(KPIEngineBase):
    mode_name = 'universal'

    def compute(self, pr, config=None):
        cfg = config or self.cfg
        res = KPIResult(mode='universal')
        df = pr.df_clean
        if df is None:
            res.warnings.append('No table data available')
            return res

        res.computed['rows'] = int(df.shape[0])
        res.computed['cols'] = int(df.shape[1])
        res.computed['numeric_columns'] = len(df.select_dtypes(include=['number']).columns)
        res.computed['categorical_columns'] = len(df.select_dtypes(include=['object','category']).columns)
        res.computed['datetime_columns'] = len(df.select_dtypes(include=['datetime','datetimetz']).columns)
        res.computed['missing_fraction_overall'] = float(df.isna().mean().mean())
        res.computed['duplicate_rows'] = int(df.duplicated().sum())
        # constant columns
        nun = df.nunique(dropna=False)
        const = [c for c, v in nun.items() if v <= 1]
        res.computed['constant_columns'] = const

        # correlations
        num = df.select_dtypes(include=['number'])
        if num.shape[1] >= 2:
            try:
                corr = num.corr().abs()
                # find strongest pair
                corr_vals = corr.where(~np.eye(corr.shape[0],dtype=bool)).stack().sort_values(ascending=False)
                top = corr_vals.head(5).to_dict()
                res.tables['top_correlations'] = [{ 'pair': k, 'corr': float(v) } for k,v in top.items()]
                # heatmap chart
                res.charts.append({
                    'id': 'universal_corr_heatmap',
                    'chart_type': 'heatmap',
                    'title': 'Correlation Heatmap',
                    'priority': 1,
                })
            except Exception:
                pass

        # skewed numeric
        sk = {}
        for c in num.columns:
            try:
                s = num[c].dropna()
                if len(s) > 0:
                    sk[c] = float(s.skew())
            except Exception:
                continue
        if sk:
            # top absolute skew
            top_sk = sorted(sk.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            res.tables['top_skewed'] = [{ 'column': k, 'skew': v } for k,v in top_sk]
            # hist for most skewed
            if top_sk:
                col0 = top_sk[0][0]
                res.charts.append({
                    'id': f'universal_hist_{col0}',
                    'chart_type': 'hist',
                    'title': f'Distribution: {col0}',
                    'y': col0,
                    'priority': 3,
                })

        # plain english
        res.summary = f"Dataset: {res.computed['rows']} rows × {res.computed['cols']} cols. {len(const)} constant columns dropped. {res.computed['numeric_columns']} numeric columns."
        return res
