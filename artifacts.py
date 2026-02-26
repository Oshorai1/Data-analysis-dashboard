from pathlib import Path
import json
from dataclasses import asdict
from typing import Dict, Any

def make_run_dirs(base: str, run_id: str) -> Dict[str, str]:
    basep = Path(base) / run_id
    charts = basep / 'charts'
    basep.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    return { 'output_dir': str(basep), 'charts_dir': str(charts) }

def write_report(report: Dict[str, Any], output_dir: str) -> None:
    p = Path(output_dir)
    with open(p / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    # Markdown with KPIs and charts
    md_lines = []
    md_lines.append(f"# Report\n")
    md_lines.append(f"**Run ID:** {report.get('run_id')}\n")
    md_lines.append(f"**Mode:** {report.get('mode')}\n")
    md_lines.append("## KPI Summary\n")
    kpis = report.get('kpis', {})
    if kpis:
        md_lines.append(kpis.get('summary','') or '')
        md_lines.append('\n### Key KPIs\n')
        comp = kpis.get('computed', {})
        for k, v in comp.items():
            md_lines.append(f"- **{k}**: {v}")
        # tables
        for tname, tbl in (kpis.get('tables') or {}).items():
            md_lines.append(f"\n### {tname}\n")
            md_lines.append('```json')
            try:
                md_lines.append(json.dumps(tbl[:200], indent=2, default=str))
            except Exception:
                md_lines.append(str(tbl)[:200])
            md_lines.append('```')

    # charts list
    md_lines.append('\n## Charts\n')
    for c in (report.get('charts') or []):
        md_lines.append(f"- {c.get('title')} — {c.get('filename')}")

    (p / 'report.md').write_text('\n'.join(md_lines), encoding='utf-8')

def save_csv(df, path: str) -> None:
    df.to_csv(path, index=False)
