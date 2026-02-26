from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from .utils import stable_chart_id, chart_filename_from_id, sample_fractionally
import os


class ChartRenderer:
    def __init__(self, output_dir: str, max_samples: int = 2000):
        self.output_dir = Path(output_dir)
        self.charts_dir = self.output_dir / 'charts'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = int(max_samples)

    def render_all(self, pipeline_result, kpi_result, analysis_results: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        saved: List[Dict[str, Any]] = []
        seen_ids = set()
        charts = (kpi_result.charts or []) if kpi_result else []

        df = pipeline_result.df_clean
        n_rows = len(df) if df is not None else 0

        for desc in charts:
            try:
                cid = stable_chart_id(desc, kpi_result.mode if kpi_result else 'unknown')
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                fname = chart_filename_from_id(cid)
                out_path = self.charts_dir / fname
                # deterministic: skip rendering if file exists
                if out_path.exists():
                    saved.append({'id': cid, 'filename': fname, 'title': desc.get('title'), 'chart_type': desc.get('chart_type')})
                    continue

                ctype = (desc.get('chart_type') or '').lower()
                # sample df if large
                frac = sample_fractionally(n_rows, self.max_samples)
                dsub = df.sample(frac=frac, random_state=42) if (df is not None and frac < 1.0) else df

                ok = False
                if ctype == 'hist' and dsub is not None and desc.get('y'):
                    ok = self._render_hist(dsub, desc, out_path)
                elif ctype in ('line','time_series') and dsub is not None and desc.get('x') and desc.get('y'):
                    ok = self._render_line(dsub, desc, out_path)
                elif ctype == 'bar' and dsub is not None:
                    ok = self._render_bar(dsub, desc, out_path)
                elif ctype == 'heatmap' and dsub is not None:
                    ok = self._render_heatmap(dsub, desc, out_path)
                elif ctype == 'scatter' and dsub is not None and desc.get('x') and desc.get('y'):
                    ok = self._render_scatter(dsub, desc, out_path)
                elif ctype == 'equity_curve' and dsub is not None and desc.get('y'):
                    ok = self._render_equity(dsub, desc, out_path)
                elif ctype == 'drawdown' and dsub is not None and desc.get('y'):
                    ok = self._render_drawdown(dsub, desc, out_path)
                elif ctype == 'candlestick' and dsub is not None:
                    ok = self._render_candlestick(dsub, desc, out_path)
                else:
                    # fallback: try simple line of y
                    if dsub is not None and desc.get('y'):
                        ok = self._render_line(dsub, desc, out_path)

                if ok:
                    saved.append({'id': cid, 'filename': fname, 'title': desc.get('title'), 'chart_type': desc.get('chart_type')})
                else:
                    # create empty placeholder or skip
                    pass
            except Exception:
                continue

        return saved

    def _render_hist(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        y = desc.get('y')
        try:
            series = pd.to_numeric(df[y], errors='coerce').dropna()
            if series.empty:
                return False
            plt.figure(figsize=(6,4))
            plt.hist(series, bins=40)
            plt.title(desc.get('title') or f"Distribution: {y}")
            plt.xlabel(y)
            plt.ylabel('count')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return True
        except Exception:
            return False

    def _render_line(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        x = desc.get('x'); y = desc.get('y')
        try:
            serx = pd.to_datetime(df[x], errors='coerce') if x in df.columns else None
            sery = pd.to_numeric(df[y], errors='coerce') if y in df.columns else None
            if serx is None or sery is None:
                return False
            df2 = pd.DataFrame({'x': serx, 'y': sery}).dropna()
            if df2.empty:
                return False
            df2 = df2.sort_values('x')
            plt.figure(figsize=(8,4))
            plt.plot(df2['x'], df2['y'], '-o', markersize=2)
            plt.title(desc.get('title') or y)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            return True
        except Exception:
            return False

    def _render_bar(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        # if group_by present, aggregate y by group
        try:
            group = desc.get('group_by')
            y = desc.get('y')
            if group and y and group in df.columns and y in df.columns:
                agg = df.groupby(group)[y].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()).sort_values(ascending=False).head(20)
                if agg.empty:
                    return False
                plt.figure(figsize=(8,4))
                plt.bar(agg.index.astype(str), agg.values)
                plt.title(desc.get('title') or y)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(out_path)
                plt.close()
                return True
            # else if descriptor includes a table list in desc['data']
            data = desc.get('data')
            if isinstance(data, list) and data:
                keys = list(data[0].keys())
                if len(keys) >= 2:
                    labels = [str(r[keys[0]]) for r in data[:20]]
                    vals = [float(r[keys[1]] or 0) for r in data[:20]]
                    plt.figure(figsize=(8,4))
                    plt.bar(labels, vals)
                    plt.xticks(rotation=45, ha='right')
                    plt.title(desc.get('title') or '')
                    plt.tight_layout(); plt.savefig(out_path); plt.close(); return True
            return False
        except Exception:
            return False

    def _render_heatmap(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        try:
            num = df.select_dtypes(include=['number'])
            if num.shape[1] < 2:
                return False
            corr = num.corr()
            plt.figure(figsize=(8,6))
            plt.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title(desc.get('title') or 'Correlation')
            plt.tight_layout(); plt.savefig(out_path); plt.close(); return True
        except Exception:
            return False

    def _render_scatter(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        x = desc.get('x'); y = desc.get('y')
        try:
            if x not in df.columns or y not in df.columns:
                return False
            xx = pd.to_numeric(df[x], errors='coerce'); yy = pd.to_numeric(df[y], errors='coerce')
            df2 = pd.DataFrame({'x': xx, 'y': yy}).dropna()
            if df2.empty:
                return False
            plt.figure(figsize=(6,6))
            plt.scatter(df2['x'], df2['y'], s=8)
            plt.xlabel(x); plt.ylabel(y)
            plt.title(desc.get('title') or f'{y} vs {x}')
            plt.tight_layout(); plt.savefig(out_path); plt.close(); return True
        except Exception:
            return False

    def _render_equity(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        y = desc.get('y')
        try:
            if y not in df.columns:
                return False
            series = pd.to_numeric(df[y], errors='coerce').dropna()
            if series.empty:
                return False
            plt.figure(figsize=(10,4))
            plt.plot(series.values)
            plt.title(desc.get('title') or 'Equity Curve')
            plt.xlabel('index'); plt.ylabel(y)
            plt.tight_layout(); plt.savefig(out_path); plt.close(); return True
        except Exception:
            return False

    def _render_drawdown(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        y = desc.get('y')
        try:
            if y not in df.columns:
                return False
            series = pd.to_numeric(df[y], errors='coerce').fillna(method='ffill').dropna()
            if series.empty:
                return False
            cum = np.cumsum(series) if len(series)>0 else series
            high = np.maximum.accumulate(cum)
            drawdown = cum - high
            plt.figure(figsize=(10,4))
            plt.plot(drawdown)
            plt.title(desc.get('title') or 'Drawdown')
            plt.tight_layout(); plt.savefig(out_path); plt.close(); return True
        except Exception:
            return False

    def _render_candlestick(self, df: pd.DataFrame, desc: Dict[str, Any], out_path: Path) -> bool:
        # Expect columns: open, high, low, close and optional datetime
        try:
            o = desc.get('open') or 'open'
            h = desc.get('high') or 'high'
            l = desc.get('low') or 'low'
            c = desc.get('close') or 'close'
            dt = desc.get('x')
            if not all(k in df.columns for k in (o,h,l,c)):
                return False
            sub = df[[o,h,l,c]].copy()
            if dt and dt in df.columns:
                sub[dt] = pd.to_datetime(df[dt], errors='coerce')
                sub = sub.dropna(subset=[dt])
                sub = sub.sort_values(dt).head(200)
            else:
                sub = sub.head(200)

            fig, ax = plt.subplots(figsize=(10,5))
            width = 0.6
            for idx, row in enumerate(sub.itertuples(index=False)):
                open_p = getattr(row, o)
                high_p = getattr(row, h)
                low_p = getattr(row, l)
                close_p = getattr(row, c)
                color = 'green' if close_p >= open_p else 'red'
                ax.vlines(idx, low_p, high_p, color='black', linewidth=0.5)
                rect_bottom = min(open_p, close_p)
                rect_height = abs(close_p - open_p)
                ax.add_patch(plt.Rectangle((idx - width/2, rect_bottom), width, rect_height if rect_height>0 else 0.0001, color=color))
            ax.set_title(desc.get('title') or 'Candlestick')
            ax.set_xlim(-1, len(sub)+1)
            plt.tight_layout(); plt.savefig(out_path); plt.close(); return True
        except Exception:
            return False
