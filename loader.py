from pathlib import Path
from typing import Any, Dict
import pandas as pd
import logging
from .utils import detect_encoding, detect_delimiter
from .errors import LoadError

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False

log = logging.getLogger("pipeline.loader")

def load(path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise LoadError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext == ".csv" or cfg.get("type_override") == "csv":
        return _load_csv(p, cfg)
    if ext in (".xls", ".xlsx") or cfg.get("type_override") == "excel":
        return _load_excel(p, cfg)
    if ext == ".parquet" or cfg.get("type_override") == "parquet":
        return _load_parquet(p, cfg)
    if ext == ".json" or cfg.get("type_override") == "json":
        return _load_json(p, cfg)
    if ext == ".pdf" or cfg.get("type_override") == "pdf":
        return _load_pdf(p, cfg)
    if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp") or cfg.get("type_override") == "image":
        return _load_image(p, cfg)

    # fallback: try CSV then text
    try:
        return _load_csv(p, cfg)
    except Exception:
        raise LoadError(f"Unsupported file type or failed to load: {ext}")

def _load_csv(p: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    enc = detect_encoding(str(p))
    delim_guess = detect_delimiter(str(p), enc)

    read_attempts = []
    # candidate delimiters: start with sniffer guess then common others
    candidates = [delim_guess] + [d for d in [',', '\t', ';', '|'] if d != delim_guess]

    def _clean_columns(df):
        # header repair: strip, remove BOM, replace illegal chars, dedupe
        new_cols = []
        seen = {}
        for c in list(df.columns):
            if isinstance(c, bytes):
                try:
                    c = c.decode('utf-8', errors='ignore')
                except Exception:
                    c = str(c)
            name = str(c).strip()
            # remove BOM
            if name.startswith('\ufeff'):
                name = name.lstrip('\ufeff')
            # replace problematic characters
            name = name.replace('\n',' ').replace('\r',' ').strip()
            name = ''.join([ch if (ch.isalnum() or ch in ['_','-',' ']) else '_' for ch in name])
            if not name:
                name = 'col'
            base = name
            i = 1
            while name in seen:
                i += 1
                name = f"{base}__{i}"
            seen[name] = True
            new_cols.append(name)
        df.columns = new_cols
        return df

    last_exc = None
    best_df = None
    best_cols = 0
    best_delim = None
    
    for delim in candidates:
        try:
            # try fast C engine first
            df = pd.read_csv(str(p), encoding=enc, delimiter=delim, engine='c', low_memory=False, on_bad_lines='skip')
            # if pandas collapsed everything into one column, try next delimiter
            if df.shape[1] <= 1:
                # try python engine with liberal quoting
                try:
                    df2 = pd.read_csv(str(p), encoding=enc, delimiter=delim, engine='python', quoting=3, low_memory=False, on_bad_lines='skip')
                    if df2.shape[1] > df.shape[1]:
                        df = df2
                except Exception:
                    pass
            
            # track best result so far
            if df.shape[1] > best_cols:
                best_df = df
                best_cols = df.shape[1]
                best_delim = delim
                
            if df.shape[1] > 1:
                # found a good parse, keep it
                df = _clean_columns(df)
                log.info(f"CSV loaded with delimiter='{repr(delim)}' encoding={enc}")
                return {"kind": "table", "df": df, "source_type": "csv"}
        except Exception as e:
            last_exc = e
            log.debug(f"CSV parse failed for delimiter {repr(delim)}: {e}")
    
    # fallback: use best result even if it's only 1 column
    if best_df is not None:
        best_df = _clean_columns(best_df)
        log.warning(f"CSV loaded with best-effort delimiter='{repr(best_delim)}' (only {best_cols} cols)")
        return {"kind": "table", "df": best_df, "source_type": "csv"}

    # final fallback: try pandas default with python engine and infer separator via regex
    try:
        df = pd.read_csv(str(p), encoding=enc, engine='python', sep=None, low_memory=False, on_bad_lines='skip')
        df = _clean_columns(df)
        return {"kind": "table", "df": df, "source_type": "csv"}
    except Exception as e:
        log.exception("CSV load failed")
        raise LoadError(f"CSV load failed: {last_exc or e}")

def _load_excel(p: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sheet = cfg.get("excel_sheet")
        df = pd.read_excel(str(p), sheet_name=sheet)
        return {"kind": "table", "df": df, "source_type": "excel"}
    except Exception as e:
        log.exception("Excel load failed")
        raise LoadError(f"Excel load failed: {e}")

def _load_parquet(p: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = pd.read_parquet(str(p))
        return {"kind": "table", "df": df, "source_type": "parquet"}
    except Exception as e:
        log.exception("Parquet load failed")
        raise LoadError(f"Parquet load failed: {e}")

def _load_json(p: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        obj = pd.read_json(str(p), lines=False)
        if hasattr(obj, 'columns'):
            return {"kind": "table", "df": obj, "source_type": "json"}
        else:
            return {"kind": "text", "text": str(obj), "source_type": "json"}
    except Exception as e:
        log.exception("JSON load failed")
        raise LoadError(f"JSON load failed: {e}")

def _load_pdf(p: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    text_parts = []
    if HAS_PYPDF:
        try:
            reader = PdfReader(str(p))
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n".join([t for t in text_parts if t.strip()])
            return {"kind": "document", "text": text, "source_type": "pdf"}
        except Exception as e:
            log.exception("PDF load failed (pypdf)")
            raise LoadError(f"PDF load failed: {e}")
    else:
        raise LoadError("pypdf not available for PDF extraction")

def _load_image(p: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not HAS_OCR or not cfg.get("allow_ocr", True):
        raise LoadError("OCR not available or disabled")
    try:
        img = Image.open(str(p))
        txt = pytesseract.image_to_string(img)
        return {"kind": "document", "text": txt, "source_type": "image"}
    except Exception as e:
        log.exception("Image OCR failed")
        raise LoadError(f"Image OCR failed: {e}")
