from pathlib import Path
import csv

try:
    import chardet
    HAS_CHARDET = True
except Exception:
    HAS_CHARDET = False

def detect_encoding(path: str, sample_size: int = 4096) -> str:
    try:
        if HAS_CHARDET:
            with open(path, "rb") as f:
                raw = f.read(sample_size)
            res = chardet.detect(raw)
            enc = res.get("encoding") or "utf-8"
            return enc
        else:
            return "utf-8"
    except Exception:
        return "utf-8"

def detect_delimiter(path: str, encoding: str = "utf-8") -> str:
    # sniff using csv.Sniffer and fallback heuristics
    try:
        with open(path, "r", encoding=encoding, errors="ignore") as f:
            sample = f.read(16384)
            sniffer = csv.Sniffer()
            try:
                delim = sniffer.sniff(sample).delimiter
                # sometimes sniffer returns '\r' or unexpected - guard
                if delim and isinstance(delim, str) and delim.strip():
                    return delim
            except Exception:
                pass

            # fallback: test common delimiters and pick the one with most columns
            candidates = [',', '\t', ';', '|']
            best = ','
            best_score = -1
            lines = [l for l in sample.splitlines() if l.strip()][:20]
            if not lines:
                return ','
            for d in candidates:
                try:
                    counts = [len(l.split(d)) for l in lines]
                    # score prefers more columns and low variance
                    score = (sum(counts)/len(counts)) - (max(counts)-min(counts))*0.1
                    if score > best_score:
                        best_score = score
                        best = d
                except Exception:
                    continue
            return best
    except Exception:
        return ','
