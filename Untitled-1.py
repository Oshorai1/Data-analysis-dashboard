"""Universal Analytics Engine (v5)

Single clean build (NO duplication) that supports:
- Structured: CSV, Excel, Parquet, SQLite
- Semi-structured: JSON, JSONL, logs
- Unstructured: TXT, PDF (text), Images (OCR if available)

Tasks (knobs):
- Descriptive
- Classification
- Regression
- Clustering
- Anomaly
- NLP summary (basic)

Outputs:
- outputs/<run_id>/report.json (source of truth)
- outputs/<run_id>/report.md (human readable)
- outputs/<run_id>/charts/*.png

Run modes:
1) CLI (recommended):
   python universal_analytics_v5.py --config config.yaml

2) Quick run:
   python universal_analytics_v5.py --input data.csv --run descriptive

NOTE:
- OCR and PDF table extraction are best-effort.
- If optional dependencies are missing, the engine degrades gracefully.

Author: Osho Rai
"""

from __future__ import annotations

import os
import re
import io
import json
import uuid
import math
import time
import sqlite3
import logging
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Matplotlib only (no seaborn). This file is CLI-friendly.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest


# -----------------------------
# Optional Dependencies (best-effort)
# -----------------------------

HAS_YAML = False
try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:
    HAS_YAML = False

HAS_PYPDF = False
try:
    from pypdf import PdfReader  # type: ignore
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

HAS_PIL = False
try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    HAS_PIL = False

HAS_TESSERACT = False
try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False


# Optional trading KPIs module
HAS_TRADING = False
try:
    from analytics.modes.trading import compute_trading_kpis  # type: ignore
    HAS_TRADING = True
except Exception:
    HAS_TRADING = False


# -----------------------------
# Defaults (single source of truth)
# -----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "input": {
        "path": "",
        "type_override": "",  # csv|excel|parquet|sqlite|json|jsonl|txt|pdf|image
        "encoding": "utf-8",
        "max_rows": 250000,
        "csv_chunk_size": 50000,
        "excel_sheet": None,
        "sqlite_table": None,
    },
    "run": {
        "descriptive": True,
        "classification": False,
        "regression": False,
        "clustering": False,
        "anomaly": False,
        "nlp_summary": True,
        "all": False,
    },
    "targets": {
        "classification_target": "",
        "regression_target": "",
        "text_column": "",  # for NLP summary or text modeling
    },
    "preprocess": {
        "drop_high_cardinality_threshold": 200,
        "drop_constant_columns": True,
        "max_onehot_categories": 50,
        "numeric_outlier_clip": True,
        "outlier_clip_quantiles": [0.01, 0.99],
    },
    "models": {
        "random_state": 42,
        "test_size": 0.2,
        "classification_model": "random_forest",  # random_forest|logreg
        "regression_model": "random_forest",      # random_forest|ridge
        "rf_n_estimators": 250,
        "rf_max_depth": None,
        "logreg_max_iter": 2000,
        "ridge_alpha": 1.0,
        "kmeans_k": 5,
        "pca_components": 2,
        "anomaly_contamination": 0.02,
        "tfidf_max_features": 8000,
        "tfidf_ngram_range": [1, 2],
    },
    "ocr": {
        "enabled": True,
        "tesseract_cmd": "",  # set if needed
    },
    "report": {
        "output_dir": "outputs",
        "save_charts": True,
        "max_top_features": 15,
    },
    "mode": "universal",
    "logging": {
        "level": "INFO",
    },
}


# -----------------------------
# Utilities
# -----------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _safe_mkdir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x


# -----------------------------
# Config Loader
# -----------------------------

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if not config_path:
        return cfg

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    text = p.read_text(encoding="utf-8", errors="ignore")

    if p.suffix.lower() in (".yaml", ".yml"):
        if not HAS_YAML:
            raise RuntimeError(
                "PyYAML is not installed. Install with: pip install pyyaml"
            )
        user_cfg = yaml.safe_load(text) or {}
    elif p.suffix.lower() == ".json":
        user_cfg = json.loads(text)
    else:
        raise ValueError("Config must be .yaml/.yml or .json")

    return _deep_merge(cfg, user_cfg)


# -----------------------------
# Document / File Ingestion
# -----------------------------

class IngestionError(Exception):
    pass


class UniversalIngestor:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.log = logger

        if cfg.get("ocr", {}).get("tesseract_cmd") and HAS_TESSERACT:
            try:
                pytesseract.pytesseract.tesseract_cmd = cfg["ocr"]["tesseract_cmd"]
            except Exception:
                pass

    def infer_type(self, path: str) -> str:
        override = (self.cfg.get("input", {}).get("type_override") or "").strip().lower()
        if override:
            return override

        ext = Path(path).suffix.lower()
        if ext in (".csv",):
            return "csv"
        if ext in (".xlsx", ".xls"):
            return "excel"
        if ext in (".parquet",):
            return "parquet"
        if ext in (".db", ".sqlite", ".sqlite3"):
            return "sqlite"
        if ext in (".json",):
            return "json"
        if ext in (".jsonl", ".ndjson"):
            return "jsonl"
        if ext in (".txt", ".log", ".md"):
            return "txt"
        if ext in (".pdf",):
            return "pdf"
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
            return "image"

        # fallback: treat as text
        return "txt"

    def load(self, path: str) -> Dict[str, Any]:
        path = str(path)
        if not path:
            raise IngestionError("No input path provided")
        if not Path(path).exists():
            raise IngestionError(f"Input path does not exist: {path}")

        t = self.infer_type(path)
        self.log.info(f"Ingesting file: type={t} path={path}")

        if t == "csv":
            df = self._load_csv(path)
            return {"kind": "table", "df": df, "source_type": t}

        if t == "excel":
            df = self._load_excel(path)
            return {"kind": "table", "df": df, "source_type": t}

        if t == "parquet":
            df = self._load_parquet(path)
            return {"kind": "table", "df": df, "source_type": t}

        if t == "sqlite":
            df = self._load_sqlite(path)
            return {"kind": "table", "df": df, "source_type": t}

        if t == "json":
            df = self._load_json(path)
            return {"kind": "table", "df": df, "source_type": t}

        if t == "jsonl":
            df = self._load_jsonl(path)
            return {"kind": "table", "df": df, "source_type": t}

        if t == "txt":
            text = self._load_text(path)
            return {"kind": "text", "text": text, "source_type": t}

        if t == "pdf":
            extracted = self._load_pdf(path)
            return {"kind": "document", **extracted, "source_type": t}

        if t == "image":
            extracted = self._load_image(path)
            return {"kind": "document", **extracted, "source_type": t}

        # fallback
        text = self._load_text(path)
        return {"kind": "text", "text": text, "source_type": t}

    def _load_csv(self, path: str) -> pd.DataFrame:
        enc = self.cfg["input"].get("encoding", "utf-8")
        max_rows = int(self.cfg["input"].get("max_rows", 250000))
        chunk_size = int(self.cfg["input"].get("csv_chunk_size", 50000))

        try:
            chunks = []
            read_rows = 0
            for chunk in pd.read_csv(path, encoding=enc, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                read_rows += len(chunk)
                if read_rows >= max_rows:
                    break
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            return df
        except Exception as e:
            raise IngestionError(f"CSV load failed: {e}")

    def _load_excel(self, path: str) -> pd.DataFrame:
        sheet = self.cfg["input"].get("excel_sheet")
        max_rows = int(self.cfg["input"].get("max_rows", 250000))
        try:
            df = pd.read_excel(path, sheet_name=sheet)
            if len(df) > max_rows:
                df = df.head(max_rows)
            return df
        except Exception as e:
            raise IngestionError(f"Excel load failed: {e}")

    def _load_parquet(self, path: str) -> pd.DataFrame:
        max_rows = int(self.cfg["input"].get("max_rows", 250000))
        try:
            df = pd.read_parquet(path)
            if len(df) > max_rows:
                df = df.head(max_rows)
            return df
        except Exception as e:
            raise IngestionError(f"Parquet load failed: {e}")

    def _load_sqlite(self, path: str) -> pd.DataFrame:
        table = self.cfg["input"].get("sqlite_table")
        max_rows = int(self.cfg["input"].get("max_rows", 250000))
        try:
            con = sqlite3.connect(path)
            try:
                if not table:
                    # choose first table
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con
                    )
                    if tables.empty:
                        raise IngestionError("SQLite DB has no tables")
                    table = str(tables.iloc[0, 0])

                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {max_rows}", con)
                return df
            finally:
                con.close()
        except Exception as e:
            raise IngestionError(f"SQLite load failed: {e}")

    def _load_json(self, path: str) -> pd.DataFrame:
        max_rows = int(self.cfg["input"].get("max_rows", 250000))
        try:
            obj = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, list):
                df = pd.json_normalize(obj)
            elif isinstance(obj, dict):
                # try best effort
                if "data" in obj and isinstance(obj["data"], list):
                    df = pd.json_normalize(obj["data"])
                else:
                    df = pd.json_normalize([obj])
            else:
                df = pd.DataFrame({"value": [str(obj)]})

            if len(df) > max_rows:
                df = df.head(max_rows)
            return df
        except Exception as e:
            raise IngestionError(f"JSON load failed: {e}")

    def _load_jsonl(self, path: str) -> pd.DataFrame:
        max_rows = int(self.cfg["input"].get("max_rows", 250000))
        try:
            rows = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= max_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        rows.append({"raw": line})
            df = pd.json_normalize(rows) if rows else pd.DataFrame()
            return df
        except Exception as e:
            raise IngestionError(f"JSONL load failed: {e}")

    def _load_text(self, path: str) -> str:
        enc = self.cfg["input"].get("encoding", "utf-8")
        try:
            return Path(path).read_text(encoding=enc, errors="ignore")
        except Exception:
            # fallback
            return Path(path).read_text(encoding="utf-8", errors="ignore")

    def _load_pdf(self, path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text": "", "tables": []}

        if HAS_PYPDF:
            try:
                reader = PdfReader(path)
                parts = []
                for page in reader.pages:
                    try:
                        parts.append(page.extract_text() or "")
                    except Exception:
                        continue
                out["text"] = "\n".join([p for p in parts if p.strip()])
            except Exception as e:
                self.log.warning(f"PDF text extraction failed (pypdf): {e}")
        else:
            self.log.warning("pypdf not installed; PDF text extraction unavailable")

        # OCR fallback: only if text is empty
        if (not out["text"].strip()) and self.cfg.get("ocr", {}).get("enabled", True):
            if HAS_TESSERACT and HAS_PIL:
                self.log.warning(
                    "PDF appears image-based. OCR fallback is not implemented in pure python here. "
                    "(Requires pdf2image + poppler)."
                )
            else:
                self.log.warning("OCR not available (pytesseract/PIL missing)")

        return out

    def _load_image(self, path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text": ""}

        if not self.cfg.get("ocr", {}).get("enabled", True):
            return out

        if not (HAS_PIL and HAS_TESSERACT):
            self.log.warning("OCR not available (install pillow + pytesseract)")
            return out

        try:
            img = Image.open(path)
            txt = pytesseract.image_to_string(img)
            out["text"] = txt
            return out
        except Exception as e:
            self.log.warning(f"Image OCR failed: {e}")
            return out


# -----------------------------
# Core Analytics Engine
# -----------------------------

@dataclass
class RunArtifacts:
    run_id: str
    output_dir: str
    charts_dir: str
    report_json_path: str
    report_md_path: str


class UniversalAnalyticsEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        level = (cfg.get("logging", {}).get("level") or "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        self.log = logging.getLogger("universal_analytics_v5")

        self.ingestor = UniversalIngestor(cfg, self.log)

    # Expose supported tasks and models so external UIs can query capabilities
    def get_supported_tasks(self) -> List[str]:
        return list(self.cfg.get("run", {}).keys())

    def get_supported_models(self) -> List[str]:
        return list(self.cfg.get("models", {}).keys())

    def _resolve_run_knobs(self) -> Dict[str, bool]:
        r = dict(self.cfg.get("run", {}))
        if r.get("all"):
            return {
                "descriptive": True,
                "classification": True,
                "regression": True,
                "clustering": True,
                "anomaly": True,
                "nlp_summary": True,
            }
        return {
            "descriptive": bool(r.get("descriptive", True)),
            "classification": bool(r.get("classification", False)),
            "regression": bool(r.get("regression", False)),
            "clustering": bool(r.get("clustering", False)),
            "anomaly": bool(r.get("anomaly", False)),
            "nlp_summary": bool(r.get("nlp_summary", True)),
        }

    def _make_artifacts(self) -> RunArtifacts:
        run_id = _now_id()
        base = _safe_mkdir(Path(self.cfg["report"].get("output_dir", "outputs")) / run_id)
        charts = _safe_mkdir(base / "charts")
        return RunArtifacts(
            run_id=run_id,
            output_dir=str(base),
            charts_dir=str(charts),
            report_json_path=str(base / "report.json"),
            report_md_path=str(base / "report.md"),
        )

    def run(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        artifacts = self._make_artifacts()
        knobs = self._resolve_run_knobs()

        if input_path:
            self.cfg["input"]["path"] = input_path

        path = self.cfg["input"].get("path")
        if not path:
            raise ValueError("No input.path provided")

        raw = self.ingestor.load(path)

        report: Dict[str, Any] = {
            "run_id": artifacts.run_id,
            "created_at": datetime.now().isoformat(),
            "mode": self.cfg.get("mode", "universal"),
            "input": {
                "path": path,
                "source_type": raw.get("source_type"),
                "kind": raw.get("kind"),
            },
            "knobs": knobs,
            "warnings": [],
            "results": {},
            "tasks_requested": [k for k, v in knobs.items() if v],
            "tasks_ran": [],
            "tasks_skipped": {},
            "inferred_types": {},
        }

        try:
            if raw["kind"] == "table":
                df = raw["df"]
                # infer column types and include in report
                try:
                    inferred = self._infer_column_types(df)
                    report["inferred_types"] = inferred
                except Exception:
                    report["warnings"].append("Column inference failed")

                table_results = self._run_table_pipeline(df, artifacts, knobs)
                report["results"].update(table_results)
                # record tasks ran/skipped based on keys present in results
                for t in report.get("tasks_requested", []):
                    if t in report["results"]:
                        report["tasks_ran"].append(t)
                    else:
                        report["tasks_skipped"][t] = "Not executed or missing requirements"

            elif raw["kind"] in ("text", "document"):
                text = raw.get("text", "")
                report["results"].update(self._run_text_pipeline(text, artifacts, knobs))

            else:
                report["warnings"].append(f"Unknown input kind: {raw['kind']}")

        except Exception as e:
            tb = traceback.format_exc()
            report["results"]["fatal_error"] = str(e)
            report["results"]["traceback"] = tb

        self._write_reports(report, artifacts)
        self.log.info(f"DONE. Outputs: {artifacts.output_dir}")
        return report

    # -----------------------------
    # Table pipeline
    # -----------------------------

    def _run_table_pipeline(self, df: pd.DataFrame, artifacts: RunArtifacts, knobs: Dict[str, bool]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if df is None or df.empty:
            return {"warning": "Empty table"}

        # sanitize columns
        df = df.copy()
        df.columns = [self._clean_col(c) for c in df.columns]

        # --- datetime detection & parsing ---
        dt_candidates: List[str] = []
        name_re = re.compile(r"(^|_)((ts|time|timestamp|date|datetime|dt))($|_)", flags=re.I)
        for c in df.columns:
            if name_re.search(c.lower()):
                # try parsing
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().sum() > 0:
                        df[c] = parsed
                        dt_candidates.append(c)
                        continue
                except Exception:
                    pass
        # numeric unix timestamp heuristics for numeric columns
        for c in df.select_dtypes(include=["number"]).columns:
            if c in dt_candidates:
                continue
            vals = df[c].dropna()
            if vals.empty:
                continue
            # check scale
            vmin = vals.min()
            vmax = vals.max()
            # plausible unix seconds
            if vmin > 1e9 and vmax > 1e9 and vmax < 1e11:
                try:
                    parsed = pd.to_datetime(df[c], unit='s', errors='coerce')
                    if parsed.notna().sum() > 0:
                        df[c] = parsed
                        dt_candidates.append(c)
                        continue
                except Exception:
                    pass
            # plausible unix milliseconds
            if vmin > 1e12 and vmax > 1e12:
                try:
                    parsed = pd.to_datetime(df[c], unit='ms', errors='coerce')
                    if parsed.notna().sum() > 0:
                        df[c] = parsed
                        dt_candidates.append(c)
                        continue
                except Exception:
                    pass

        if dt_candidates:
            out['datetime_columns'] = dt_candidates

        # drop constant columns
        if self.cfg["preprocess"].get("drop_constant_columns", True):
            nun = df.nunique(dropna=False)
            const_cols = [c for c in df.columns if nun.get(c, 0) <= 1]
            if const_cols:
                df = df.drop(columns=const_cols, errors="ignore")
                out["dropped_constant_columns"] = const_cols

        # basic descriptive
        if knobs.get("descriptive"):
            out["descriptive"] = self._descriptive(df, artifacts)

        # anomaly (numeric only)
        if knobs.get("anomaly"):
            out["anomaly"] = self._anomaly(df, artifacts)

        # clustering
        if knobs.get("clustering"):
            out["clustering"] = self._clustering(df, artifacts)

        # supervised tasks
        cls_target = (self.cfg.get("targets", {}).get("classification_target") or "").strip()
        reg_target = (self.cfg.get("targets", {}).get("regression_target") or "").strip()

        if knobs.get("classification"):
            if not cls_target:
                out["classification"] = {"skipped": "No classification_target provided"}
            elif cls_target not in df.columns:
                out["classification"] = {"skipped": f"Target column not found: {cls_target}"}
            else:
                out["classification"] = self._classification(df, cls_target, artifacts)

        if knobs.get("regression"):
            if not reg_target:
                out["regression"] = {"skipped": "No regression_target provided"}
            elif reg_target not in df.columns:
                out["regression"] = {"skipped": f"Target column not found: {reg_target}"}
            else:
                out["regression"] = self._regression(df, reg_target, artifacts)

        # NLP summary from a text column if provided
        if knobs.get("nlp_summary"):
            text_col = (self.cfg.get("targets", {}).get("text_column") or "").strip()
            if text_col and text_col in df.columns:
                try:
                    out["nlp_summary"] = self._nlp_summary_from_column(df, text_col, artifacts)
                except Exception as e:
                    out["nlp_summary"] = {"error": str(e)}
            else:
                out["nlp_summary"] = {"skipped": "No valid text_column provided"}

        # Business mode: trading KPIs (populated when config mode=='trading')
        try:
            if str(self.cfg.get('mode', '')).lower() == 'trading':
                if HAS_TRADING:
                    try:
                        tr = compute_trading_kpis(df, Path(artifacts.charts_dir), self.cfg)
                        out['trading_kpis'] = tr
                    except Exception as e:
                        out['trading_kpis'] = {'error': str(e)}
                else:
                    out['trading_kpis'] = {'skipped': 'Trading module unavailable'}
        except Exception:
            # do not fail the whole pipeline for trading KPIs
            out.setdefault('trading_kpis', {'skipped': 'trading_kpis_failed'})

        return out

    def _clean_col(self, c: Any) -> str:
        s = str(c)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[^a-zA-Z0-9_\- ]+", "", s)
        s = s.replace(" ", "_")
        if not s:
            s = "col"
        return s

    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Infer column types robustly and return lists for each type.

        Returns a dict with keys:
          numeric_columns, categorical_columns, datetime_columns, text_columns, id_columns, boolean_columns
        """
        out: Dict[str, List[str]] = {
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "text_columns": [],
            "id_columns": [],
            "boolean_columns": [],
        }

        # heuristics
        id_name_re = re.compile(r"(^|_)(id|order_id|trade_id|tx_id|uid|user_id)($|_)", flags=re.I)
        dt_name_re = re.compile(r"(^|_)(ts|time|timestamp|date|datetime|dt)($|_)", flags=re.I)

        for c in df.columns:
            series = df[c]
            # cheap checks
            vals_nonnull = series.dropna()
            sval = c.lower()

            # ID-like by name or high-uniqueness
            if id_name_re.search(sval):
                out["id_columns"].append(c)
                continue

            # boolean detection: values in {0,1} or {True,False}
            unique_vals = set(list(vals_nonnull.unique()[:20])) if len(vals_nonnull) > 0 else set()
            if unique_vals and all((v in (0, 1, True, False) or str(v).lower() in ("0", "1", "true", "false")) for v in unique_vals):
                out["boolean_columns"].append(c)
                # booleans are categorical
                out["categorical_columns"].append(c)
                continue

            # date/datetime by name or content
            parsed = None
            if dt_name_re.search(sval):
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                except Exception:
                    parsed = None
                if parsed is not None and parsed.notna().sum() > 0:
                    out["datetime_columns"].append(c)
                    continue

            # numeric-like detection
            if pd.api.types.is_numeric_dtype(series):
                # check for unix timestamp scales
                if not series.dropna().empty:
                    vmin = series.min()
                    vmax = series.max()
                    # seconds
                    if vmin > 1e9 and vmax < 1e11:
                        try:
                            p = pd.to_datetime(series, unit="s", errors="coerce")
                            if p.notna().sum() > 0:
                                out["datetime_columns"].append(c)
                                continue
                        except Exception:
                            pass
                    # milliseconds
                    if vmin > 1e12 and vmax < 1e15:
                        try:
                            p = pd.to_datetime(series, unit="ms", errors="coerce")
                            if p.notna().sum() > 0:
                                out["datetime_columns"].append(c)
                                continue
                        except Exception:
                            pass

                # numeric but potentially categorical (low cardinality)
                try:
                    nunique = int(series.nunique(dropna=True))
                except Exception:
                    nunique = 0
                if nunique > 0 and nunique < 0.05 * max(1, len(series)) and nunique < 50:
                    out["categorical_columns"].append(c)
                else:
                    out["numeric_columns"].append(c)
                continue

            # text detection: length and uniqueness heuristics
            if pd.api.types.is_string_dtype(series) or series.dtype == object:
                sample = vals_nonnull.astype(str).head(200)
                avg_len = sample.map(len).mean() if not sample.empty else 0
                unique_ratio = sample.nunique() / max(1, len(sample)) if not sample.empty else 0
                if avg_len > 50 or unique_ratio > 0.7:
                    out["text_columns"].append(c)
                else:
                    out["categorical_columns"].append(c)
                continue

            # fallback: treat as categorical
            out["categorical_columns"].append(c)

        return out

    def _split_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        # Use robust inference to determine column types and exclude ids/datetimes
        inferred = self._infer_column_types(df)
        num_cols = inferred.get("numeric_columns", [])
        cat_cols = [c for c in inferred.get("categorical_columns", []) if c not in inferred.get("id_columns", [])]

        # drop very high cardinality categoricals
        thr = int(self.cfg["preprocess"].get("drop_high_cardinality_threshold", 200))
        kept_cat = []
        dropped = []
        for c in cat_cols:
            try:
                n = int(df[c].nunique(dropna=True))
            except Exception:
                n = 0
            if n > thr:
                dropped.append(c)
            else:
                kept_cat.append(c)

        if dropped:
            self.log.warning(f"Dropping high-cardinality columns: {dropped}")

        return num_cols, kept_cat

    def _make_preprocessor(self, df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
        # infer types and then produce numeric/categorical lists for preprocessing
        num_cols, cat_cols = self._split_types(df)

        max_cats = int(self.cfg["preprocess"].get("max_onehot_categories", 50))

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=max_cats)),
            ]
        )

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )
        return pre, num_cols, cat_cols

    # -----------------------------
    # Descriptive
    # -----------------------------

    def _descriptive(self, df: pd.DataFrame, artifacts: RunArtifacts) -> Dict[str, Any]:
        res: Dict[str, Any] = {}

        res["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
        res["columns"] = df.columns.tolist()

        missing = df.isna().mean().sort_values(ascending=False)
        res["missing_rate_top"] = (
            missing.head(25).reset_index().rename(columns={"index": "column", 0: "missing_rate"}).to_dict("records")
        )

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            desc = df[num_cols].describe().T
            res["numeric_describe"] = desc.reset_index().rename(columns={"index": "column"}).to_dict("records")

            if self.cfg["report"].get("save_charts", True):
                # Missing plot
                self._plot_missing(missing, artifacts)
                # Correlation plot (if enough numeric cols)
                if len(num_cols) >= 2:
                    self._plot_corr(df[num_cols], artifacts)

                # A few distributions
                for c in num_cols[:6]:
                    self._plot_hist(df[c], c, artifacts)

        cat_cols = [c for c in df.columns if c not in num_cols]
        top_cats = []
        for c in cat_cols[:10]:
            vc = df[c].astype(str).value_counts(dropna=False).head(10)
            top_cats.append({"column": c, "top_values": vc.to_dict()})
        res["categorical_top_values"] = top_cats

        return res

    def _plot_missing(self, missing: pd.Series, artifacts: RunArtifacts) -> None:
        s = missing[missing > 0].head(30)
        if s.empty:
            return
        plt.figure(figsize=(10, 6))
        plt.barh(list(reversed(s.index.tolist())), list(reversed(s.values.tolist())))
        plt.title("Missing Rate (Top 30)")
        plt.xlabel("Missing fraction")
        plt.tight_layout()
        plt.savefig(Path(artifacts.charts_dir) / "missing_rate.png")
        plt.close()

    def _plot_corr(self, df_num: pd.DataFrame, artifacts: RunArtifacts) -> None:
        corr = df_num.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(corr.values)
        plt.title("Correlation Heatmap (numeric)")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(Path(artifacts.charts_dir) / "correlation_heatmap.png")
        plt.close()

    def _plot_hist(self, s: pd.Series, name: str, artifacts: RunArtifacts) -> None:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(s.values, bins=40)
        plt.title(f"Distribution: {name}")
        plt.xlabel(name)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(Path(artifacts.charts_dir) / f"dist_{name}.png")
        plt.close()

    # -----------------------------
    # Anomaly
    # -----------------------------

    def _anomaly(self, df: pd.DataFrame, artifacts: RunArtifacts) -> Dict[str, Any]:
        num = df.select_dtypes(include=["number"]).copy()
        if num.shape[1] < 1 or len(num) < 50:
            return {"skipped": "Need >=1 numeric cols and >=50 rows"}

        # clip outliers to reduce IF instability
        if self.cfg["preprocess"].get("numeric_outlier_clip", True):
            ql, qh = self.cfg["preprocess"].get("outlier_clip_quantiles", [0.01, 0.99])
            for c in num.columns:
                lo = num[c].quantile(float(ql))
                hi = num[c].quantile(float(qh))
                num[c] = num[c].clip(lo, hi)

        # impute and scale manually so we can extract anomaly scores
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        try:
            X_imp = imputer.fit_transform(num)
            X_scaled = scaler.fit_transform(X_imp)
        except Exception:
            return {"skipped": "Failed preprocessing for anomaly detection"}

        model = IsolationForest(
            n_estimators=200,
            contamination=float(self.cfg["models"].get("anomaly_contamination", 0.02)),
            random_state=int(self.cfg["models"].get("random_state", 42)),
        )
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
        # decision_function: larger is more normal; invert so larger = more anomalous
        scores = -1.0 * model.decision_function(X_scaled)

        anomaly_idx = np.where(preds == -1)[0].tolist()
        frac = float(len(anomaly_idx) / len(num))

        # prepare top anomalies table with some key identifier columns
        candidate_keys = [c for c in df.columns if c.lower() in ("ts", "timestamp", "time", "date", "datetime", "id", "symbol", "price", "value")]
        if not candidate_keys:
            candidate_keys = list(df.columns[:3])

        top_rows = []
        # rank anomalies by score descending
        if len(anomaly_idx) > 0:
            ann_scores = [(i, float(scores[i])) for i in anomaly_idx]
            ann_scores = sorted(ann_scores, key=lambda x: x[1], reverse=True)
            for i, sc in ann_scores[:50]:
                row = {"row_index": int(i), "anomaly_score": float(sc)}
                for k in candidate_keys:
                    try:
                        row[k] = _to_jsonable(df.iloc[i][k])
                    except Exception:
                        row[k] = None
                top_rows.append(row)

        return {
            "anomaly_fraction": frac,
            "anomaly_count": int(len(anomaly_idx)),
            "top_anomalies": top_rows,
        }

    # -----------------------------
    # Clustering
    # -----------------------------

    def _clustering(self, df: pd.DataFrame, artifacts: RunArtifacts) -> Dict[str, Any]:
        pre, num_cols, cat_cols = self._make_preprocessor(df)

        k = int(self.cfg["models"].get("kmeans_k", 5))
        pca_n = int(self.cfg["models"].get("pca_components", 2))

        model = KMeans(n_clusters=k, n_init=10, random_state=int(self.cfg["models"].get("random_state", 42)))

        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("kmeans", model),
            ]
        )

        try:
            labels = pipe.fit_predict(df)
        except Exception as e:
            return {"error": str(e)}

        counts = pd.Series(labels).value_counts().sort_index().to_dict()

        # PCA plot
        if self.cfg["report"].get("save_charts", True):
            try:
                X = pre.fit_transform(df)
                X = X.toarray() if hasattr(X, "toarray") else X
                pca = PCA(n_components=pca_n, random_state=int(self.cfg["models"].get("random_state", 42)))
                Z = pca.fit_transform(X)

                if Z.shape[1] >= 2:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(Z[:, 0], Z[:, 1], s=10)
                    plt.title("Clustering projection (PCA)")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.tight_layout()
                    plt.savefig(Path(artifacts.charts_dir) / "clustering_pca.png")
                    plt.close()
            except Exception:
                pass

        return {
            "k": k,
            "cluster_counts": {str(k): int(v) for k, v in counts.items()},
            "notes": "Clusters are not labeled. Interpret by inspecting cluster-specific feature averages.",
            "representative": {},
        }

    # -----------------------------
    # Classification
    # -----------------------------

    def _classification(self, df: pd.DataFrame, target: str, artifacts: RunArtifacts) -> Dict[str, Any]:
        y = df[target]
        X = df.drop(columns=[target], errors="ignore")

        # sanity
        if y.nunique(dropna=True) < 2:
            return {"skipped": "Target has <2 classes"}

        pre, _, _ = self._make_preprocessor(X)

        model_name = str(self.cfg["models"].get("classification_model", "random_forest")).lower()
        if model_name == "logreg":
            model = LogisticRegression(
                max_iter=int(self.cfg["models"].get("logreg_max_iter", 2000)),
                n_jobs=None,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=int(self.cfg["models"].get("rf_n_estimators", 250)),
                max_depth=self.cfg["models"].get("rf_max_depth", None),
                random_state=int(self.cfg["models"].get("random_state", 42)),
                n_jobs=-1,
            )

        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("model", model),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(self.cfg["models"].get("test_size", 0.2)),
            random_state=int(self.cfg["models"].get("random_state", 42)),
            stratify=y if y.nunique() <= 20 else None,
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        # metrics
        avg = "binary" if y.nunique() == 2 else "weighted"
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, pred, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, pred, average=avg, zero_division=0)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "classes": [str(c) for c in sorted(pd.Series(y).dropna().unique().tolist())],
            "model": model_name,
        }

        # confusion matrix plot
        if self.cfg["report"].get("save_charts", True):
            try:
                cm = confusion_matrix(y_test, pred)
                plt.figure(figsize=(6, 5))
                plt.imshow(cm)
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(Path(artifacts.charts_dir) / "confusion_matrix.png")
                plt.close()
            except Exception:
                pass

        return {
            "metrics": metrics,
            "plain_english": self._explain_classification(metrics),
        }

    def _explain_classification(self, m: Dict[str, Any]) -> str:
        acc = m.get("accuracy", 0.0)
        f1 = m.get("f1", 0.0)
        return (
            f"The classifier achieved accuracy={acc:.3f} and F1={f1:.3f}. "
            "Accuracy is overall correctness, while F1 balances false alarms vs missed detections."
        )

    # -----------------------------
    # Regression
    # -----------------------------

    def _regression(self, df: pd.DataFrame, target: str, artifacts: RunArtifacts) -> Dict[str, Any]:
        y = pd.to_numeric(df[target], errors="coerce")
        X = df.drop(columns=[target], errors="ignore")

        if y.notna().sum() < 50:
            return {"skipped": "Too few numeric target rows"}

        pre, _, _ = self._make_preprocessor(X)

        model_name = str(self.cfg["models"].get("regression_model", "random_forest")).lower()
        if model_name == "ridge":
            model = Ridge(alpha=float(self.cfg["models"].get("ridge_alpha", 1.0)))
        else:
            model = RandomForestRegressor(
                n_estimators=int(self.cfg["models"].get("rf_n_estimators", 250)),
                max_depth=self.cfg["models"].get("rf_max_depth", None),
                random_state=int(self.cfg["models"].get("random_state", 42)),
                n_jobs=-1,
            )

        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("model", model),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(self.cfg["models"].get("test_size", 0.2)),
            random_state=int(self.cfg["models"].get("random_state", 42)),
        )

        # drop NaNs from target
        mask_tr = y_train.notna()
        mask_te = y_test.notna()
        X_train = X_train.loc[mask_tr]
        y_train = y_train.loc[mask_tr]
        X_test = X_test.loc[mask_te]
        y_test = y_test.loc[mask_te]

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(math.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))

        if self.cfg["report"].get("save_charts", True):
            try:
                plt.figure(figsize=(7, 6))
                plt.scatter(y_test.values, pred, s=12)
                plt.title("Regression: Actual vs Predicted")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.tight_layout()
                plt.savefig(Path(artifacts.charts_dir) / "reg_actual_vs_pred.png")
                plt.close()
            except Exception:
                pass

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "model": model_name,
        }

        return {
            "metrics": metrics,
            "plain_english": self._explain_regression(metrics),
        }

    def _explain_regression(self, m: Dict[str, Any]) -> str:
        mae = m.get("mae", 0.0)
        rmse = m.get("rmse", 0.0)
        r2 = m.get("r2", 0.0)
        return (
            f"The regression model has MAE={mae:.3f} and RMSE={rmse:.3f} (lower is better). "
            f"R²={r2:.3f} measures how much variance is explained (closer to 1 is better)."
        )

    # -----------------------------
    # Text pipeline
    # -----------------------------

    def _run_text_pipeline(self, text: str, artifacts: RunArtifacts, knobs: Dict[str, bool]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if not text or not text.strip():
            return {"warning": "No text extracted"}

        if knobs.get("descriptive"):
            out["descriptive"] = {
                "chars": int(len(text)),
                "words": int(len(re.findall(r"\w+", text))),
                "lines": int(len(text.splitlines())),
            }

        if knobs.get("nlp_summary"):
            out["nlp_summary"] = self._nlp_summary(text)

        # For unstructured text, classification/regression would require labels.
        out["notes"] = (
            "For pure text documents, supervised modeling requires a labeled dataset. "
            "This engine focuses on extraction + summarization for single-document inputs."
        )

        return out

    def _nlp_summary(self, text: str) -> Dict[str, Any]:
        # Simple extractive summary: pick top sentences by TF-IDF salience.
        # (Not an LLM summary; deterministic and offline.)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 3:
            return {"summary": text[:800]}

        max_sents = 8
        corpus = sentences

        vec = TfidfVectorizer(
            max_features=int(self.cfg["models"].get("tfidf_max_features", 8000)),
            ngram_range=tuple(self.cfg["models"].get("tfidf_ngram_range", [1, 2])),
            stop_words="english",
        )

        X = vec.fit_transform(corpus)
        # sentence salience = sum tfidf
        scores = np.asarray(X.sum(axis=1)).reshape(-1)
        idx = np.argsort(scores)[::-1][:max_sents]
        idx = sorted(idx.tolist())
        summary = " ".join([sentences[i] for i in idx])

        return {
            "summary": summary,
            "method": "extractive_tfidf",
        }

    def _nlp_summary_from_column(self, df: pd.DataFrame, col: str, artifacts: RunArtifacts) -> Dict[str, Any]:
        s = df[col].astype(str).fillna("")
        # sample to avoid huge memory
        s = s[s.str.len() > 0]
        if s.empty:
            return {"skipped": "No text in column"}

        sample = s.sample(n=min(2000, len(s)), random_state=int(self.cfg["models"].get("random_state", 42)))
        joined = "\n".join(sample.tolist())

        # top keywords
        vec = TfidfVectorizer(
            max_features=100,
            stop_words="english",
        )
        X = vec.fit_transform(sample.tolist())
        scores = np.asarray(X.sum(axis=0)).reshape(-1)
        terms = np.array(vec.get_feature_names_out())
        top_idx = np.argsort(scores)[::-1][:25]

        return {
            "column": col,
            "top_keywords": terms[top_idx].tolist(),
            "summary": self._nlp_summary(joined).get("summary", ""),
        }

    # -----------------------------
    # Reporting
    # -----------------------------

    def _write_reports(self, report: Dict[str, Any], artifacts: RunArtifacts) -> None:
        # Ensure standardized meta exists for downstream UIs
        meta = report.get('meta', {}) or {}
        # derive from descriptive if available
        desc = report.get('results', {}).get('descriptive', {})
        if desc:
            shape = desc.get('shape', {})
            cols = desc.get('columns', [])
            meta.setdefault('rows', shape.get('rows') if isinstance(shape, dict) else shape)
            meta.setdefault('cols', shape.get('cols') if isinstance(shape, dict) else (len(cols) if cols else '?'))
            meta.setdefault('columns', cols)
            meta.setdefault('numeric_columns', [r['column'] for r in desc.get('numeric_describe', [])] if desc.get('numeric_describe') else [])
            meta.setdefault('categorical_columns', [c for c in desc.get('categorical_top_values', [])])

        # top-level run_id/input/source/created_at
        meta.setdefault('run_id', report.get('run_id'))
        meta.setdefault('input', report.get('input', {}).get('path'))
        meta.setdefault('source_type', report.get('input', {}).get('source_type'))
        meta.setdefault('created_at', datetime.now().isoformat())
        # target info
        try:
            targ_req = self.cfg.get('targets', {})
            meta.setdefault('target_requested', targ_req.get('classification_target') or targ_req.get('regression_target') or None)
            # default to requested unless later overwritten by actual run
            meta.setdefault('target_used', meta.get('target_requested'))
        except Exception:
            meta.setdefault('target_requested', None)
            meta.setdefault('target_used', None)

        report['meta'] = meta

        # Artifacts listing (charts + files)
        art: Dict[str, Any] = {}
        charts = []
        files = []
        try:
            base = Path(artifacts.output_dir)
            chdir = base / 'charts'
            if chdir.exists():
                charts = sorted({p.name for p in chdir.iterdir() if p.is_file()})
            for p in base.iterdir():
                if p.is_file() and p.name not in (Path(artifacts.report_json_path).name, Path(artifacts.report_md_path).name):
                    files.append(p.name)
        except Exception:
            pass
        art['charts'] = charts
        art['files'] = files
        report['artifacts'] = art

        # JSON
        with open(artifacts.report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_to_jsonable)

        # Markdown (human report)
        md = self._make_markdown_report(report)
        Path(artifacts.report_md_path).write_text(md, encoding="utf-8")

    def _make_markdown_report(self, report: Dict[str, Any]) -> str:
        rid = report.get("run_id")
        inp = report.get("input", {})
        knobs = report.get("knobs", {})
        results = report.get("results", {})

        lines: List[str] = []
        lines.append(f"# Universal Analytics Report (v5)\n")
        lines.append(f"**Run ID:** `{rid}`\n")
        lines.append(f"**Input:** `{inp.get('path')}`\n")
        lines.append(f"**Source Type:** `{inp.get('source_type')}`\n")
        lines.append(f"**Kind:** `{inp.get('kind')}`\n")

        lines.append("## What was run (knobs)\n")
        for k, v in knobs.items():
            lines.append(f"- **{k}**: {bool(v)}")
        lines.append("")

        # Human-first summary
        lines.append("## Plain-English Summary\n")
        lines.append(self._plain_english_summary(results))
        lines.append("")

        # Task results
        lines.append("## Detailed Results\n")
        lines.append("```json")
        lines.append(json.dumps(results, indent=2, default=_to_jsonable)[:20000])
        lines.append("```")

        lines.append("\n## Charts\n")
        lines.append("Charts are saved in the `charts/` folder next to this report.")

        return "\n".join(lines)

    def _plain_english_summary(self, results: Dict[str, Any]) -> str:
        # This is intentionally simple but reliable.
        parts: List[str] = []

        desc = results.get("descriptive")
        if isinstance(desc, dict) and "shape" in desc:
            sh = desc["shape"]
            parts.append(f"- The dataset has **{sh.get('rows')} rows** and **{sh.get('cols')} columns**.")

        cls = results.get("classification")
        if isinstance(cls, dict) and "metrics" in cls:
            m = cls["metrics"]
            parts.append(
                f"- Classification performance: accuracy={m.get('accuracy', 0):.3f}, F1={m.get('f1', 0):.3f}."
            )

        reg = results.get("regression")
        if isinstance(reg, dict) and "metrics" in reg:
            m = reg["metrics"]
            parts.append(
                f"- Regression performance: MAE={m.get('mae', 0):.3f}, RMSE={m.get('rmse', 0):.3f}, R²={m.get('r2', 0):.3f}."
            )

        an = results.get("anomaly")
        if isinstance(an, dict) and "anomaly_fraction" in an:
            parts.append(
                f"- Detected ~{an.get('anomaly_fraction', 0)*100:.2f}% anomalies in numeric patterns."
            )

        nlp = results.get("nlp_summary")
        if isinstance(nlp, dict) and nlp.get("summary"):
            parts.append("- Extracted a short summary of the text content.")

        if not parts:
            return "- No major results were produced (check config knobs and target columns)."

        return "\n".join(parts)


# -----------------------------
# CLI
# -----------------------------

def _parse_args() -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser(description="Universal Analytics Engine (v5)")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml or config.json")
    p.add_argument("--input", type=str, default=None, help="Input file path override")

    # quick knobs
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Quick knob: descriptive|classification|regression|clustering|anomaly|all",
    )

    args = p.parse_args()
    return vars(args)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.get("config"))

    # quick knob override
    if args.get("run"):
        mode = str(args["run"]).strip().lower()
        cfg["run"]["all"] = False
        for k in ("descriptive", "classification", "regression", "clustering", "anomaly", "nlp_summary"):
            cfg["run"][k] = False

        if mode == "all":
            cfg["run"]["all"] = True
        elif mode in cfg["run"]:
            cfg["run"][mode] = True
            # keep summary on by default
            cfg["run"]["nlp_summary"] = True
        else:
            raise ValueError(f"Unknown --run mode: {mode}")

    eng = UniversalAnalyticsEngine(cfg)
    eng.run(input_path=args.get("input"))


if __name__ == "__main__":
    main()
