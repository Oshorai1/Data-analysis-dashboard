"""Universal Analytics Engine - Flask Web App

Endpoints:
- GET /           -> upload form
- POST /analyze   -> upload file, run analysis, redirect to results
- GET /runs/<id>  -> show results page with rendered markdown
- GET /artifacts/<id>/<file> -> serve artifact files

Features:
- File upload size limits (default 50MB)
- Markdown rendering to HTML
- Nicer HTML results layout with charts and report
"""
from __future__ import annotations

import os
import sys
import tempfile
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import traceback

from flask import Flask, request, jsonify, send_from_directory, render_template_string, redirect, url_for, session

try:
    import markdown2
    HAS_MARKDOWN = True
except Exception:
    HAS_MARKDOWN = False


# Try loading the large engine module dynamically (filename contains a dash)
BIG_ENGINE_AVAILABLE = False
BIG_ENGINE_MODULE = None
big_path = Path(__file__).with_name('Untitled-1.py')
if big_path.exists():
    try:
        spec = importlib.util.spec_from_file_location('big_engine', str(big_path))
        mod = importlib.util.module_from_spec(spec)
        # register before executing so decorators/dataclasses see the module
        import sys
        sys.modules[spec.name] = mod
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        BIG_ENGINE_MODULE = mod
        BIG_ENGINE_AVAILABLE = hasattr(mod, 'UniversalAnalyticsEngine') and hasattr(mod, 'load_config')
    except Exception:
        BIG_ENGINE_AVAILABLE = False

# Fallback to clean engine
CLEAN_ENGINE_AVAILABLE = False
try:
    from universal_analytics_engine_clean import MinimalEngine, RunConfig  # type: ignore
    CLEAN_ENGINE_AVAILABLE = True
except Exception:
    CLEAN_ENGINE_AVAILABLE = False

# Pipeline integration
try:
    from pipeline import run_pipeline, PipelineConfig  # type: ignore
    HAS_PIPELINE = True
except Exception:
    HAS_PIPELINE = False


def _get_supported_tasks() -> Dict[str, str]:
    """Return mapping of task_key -> display label from available engine configs.

    Prefers big engine DEFAULT_CONFIG if available, otherwise falls back to
    a sane default list. This keeps UI in sync with engine capabilities.
    """
    tasks: Dict[str, str] = {}
    if BIG_ENGINE_AVAILABLE and BIG_ENGINE_MODULE:
        try:
            cfg_run = getattr(BIG_ENGINE_MODULE, "DEFAULT_CONFIG", {}).get("run", {})
            for k in cfg_run.keys():
                tasks[k] = k.replace("_", " ").title()
            return tasks
        except Exception:
            pass

    if CLEAN_ENGINE_AVAILABLE:
        # Minimal fallback list; keep 'diagnostic' and 'text' available
        common = [
            "descriptive",
            "classification",
            "regression",
            "clustering",
            "anomaly",
            "diagnostic",
            "text",
            "nlp_summary",
        ]
        for k in common:
            tasks[k] = k.replace("_", " ").title()
        return tasks

    # final fallback
    for k in ["descriptive", "classification", "regression", "clustering", "anomaly", "text"]:
        tasks[k] = k.replace("_", " ").title()
    return tasks

# Configuration
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB limit
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
if not app.secret_key:
    app.secret_key = 'dev-secret-key'

@app.route('/')
def index():
    """Home page with upload form."""
    limit_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Universal Analytics Engine</title>
        <style>
            * {{ font-family: sans-serif; }}
            body {{ max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            .status {{ margin: 20px 0; padding: 10px; background: #ecf0f1; border-radius: 4px; }}
            .status.ok {{ color: #27ae60; }}
            .status.warning {{ color: #e67e22; }}
            form {{ margin: 30px 0; }}
            label {{ display: block; margin: 15px 0 5px 0; font-weight: bold; }}
            input[type="file"], select, input[type="text"] {{ 
                width: 100%; 
                padding: 8px; 
                margin: 5px 0 15px 0; 
                border: 1px solid #bdc3c7; 
                border-radius: 4px;
                box-sizing: border-box;
            }}
            button {{ 
                background: #3498db; 
                color: white; 
                padding: 12px 30px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 16px;
            }}
            button:hover {{ background: #2980b9; }}
            .info {{ color: #7f8c8d; font-size: 14px; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Universal Analytics Engine</h1>
            <div class="status {'ok' if (BIG_ENGINE_AVAILABLE or CLEAN_ENGINE_AVAILABLE) else 'warning'}">
                {'✓ Ready' if (BIG_ENGINE_AVAILABLE or CLEAN_ENGINE_AVAILABLE) else '⚠ Limited engines available'}
                {'(big engine loaded)' if BIG_ENGINE_AVAILABLE else ''} 
                {'(clean engine loaded)' if CLEAN_ENGINE_AVAILABLE else ''}
            </div>
            
            <form method="post" action="/analyze" enctype="multipart/form-data">
                <label for="file">📁 Upload Data File:</label>
                <input type="file" id="file" name="file" required accept=".csv,.json,.jsonl,.txt,.pdf,.xlsx,.parquet">
                
                <label for="tasks">🎯 Analysis Tasks (select one or more):</label>
                <select id="tasks" name="tasks" multiple size="6">
                    <!-- Options are generated client-side on preview step -->
                    <option disabled>Upload file to configure tasks</option>
                </select>

                <p class="info">After uploading the file you will be able to select tasks and choose a target column (if applicable).</p>
                <button type="submit">🚀 Upload & Configure</button>
                <p class="info">💡 Max file size: {limit_mb}MB. Supported: CSV, Excel, JSON, JSONL, PDF, Parquet, TXT</p>
            </form>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/analyze', methods=['POST'])
def analyze():
    """Two-step: upload -> configure -> run.

    If this POST contains 'confirm' we execute the analysis. Otherwise we
    save the uploaded file, inspect columns, and render a configuration page
    allowing multi-task selection and target choice.
    """
    # If this is the confirmation step, the upload already happened and we have
    # an input_path to use.
    if request.form.get('confirm'):
        input_path = request.form.get('input_path')
        if not input_path or not Path(input_path).exists():
            return jsonify({'error': 'Input file not found on server; please re-upload.'}), 400

        tasks = request.form.getlist('tasks') or [request.form.get('tasks')]
        tasks = [t for t in tasks if t]
        target = request.form.get('target') or None
        text_col = request.form.get('text_col') or None
        mode = request.form.get('mode') or 'universal'

        # Read mapping overrides from form
        mapping_overrides = {}
        for key in request.form.keys():
            if key.startswith('mapping_'):
                sem = key.replace('mapping_', '')
                mapping_overrides[sem] = request.form.get(key) or None

        # Run pipeline with user-chosen mode and mapping overrides if available
        try:
            engine_input = input_path
            pr = None
            if HAS_PIPELINE and 'run_pipeline' in globals():
                try:
                    pipeline_cfg = PipelineConfig()
                    pr = run_pipeline(input_path, Path(input_path).name, mode=mode, cfg=pipeline_cfg, mapping_overrides=mapping_overrides)
                    od = pr.artifacts.get('output_dir') if pr and pr.artifacts else None
                    if od:
                        cleaned = Path(od) / 'cleaned.csv'
                        if cleaned.exists():
                            engine_input = str(cleaned)
                except Exception as e:
                    # pipeline failed; continue to engine run but log warning
                    tb = traceback.format_exc()
                    return jsonify({'error': f'Analysis failed: pipeline error: {str(e)[:300]}', 'traceback': tb}), 500

            # Now run analytics engine as before (keeps backward compatibility)
            if BIG_ENGINE_AVAILABLE and BIG_ENGINE_MODULE:
                cfg = BIG_ENGINE_MODULE.load_config(None)
                cfg['input']['path'] = engine_input
                for k in cfg.get('run', {}):
                    cfg['run'][k] = False
                for t in tasks:
                    if t in cfg.get('run', {}):
                        cfg['run'][t] = True
                if target:
                    cfg['targets'] = cfg.get('targets', {})
                    cfg['targets']['classification_target'] = target
                    cfg['targets']['regression_target'] = target
                if text_col:
                    cfg['targets'] = cfg.get('targets', {})
                    cfg['targets']['text_column'] = text_col
                cfg['mode'] = mode
                eng = BIG_ENGINE_MODULE.UniversalAnalyticsEngine(cfg)
                report = eng.run(engine_input)
            else:
                sel = tasks[0] if tasks else 'descriptive'
                rc = RunConfig(input_path=engine_input, task=sel, text_col=text_col, target_col=target)
                try:
                    setattr(rc, 'mode', mode)
                except Exception:
                    pass
                eng = MinimalEngine(rc)
                report = eng.run()

            run_id = report.get('meta', {}).get('run_id') or report.get('run_id')
            return redirect(url_for('run_page', run_id=run_id))
        except Exception as e:
            tb = traceback.format_exc()
            return jsonify({'error': f'Analysis failed: {str(e)[:300]}', 'traceback': tb}), 500

    # --- initial upload/inspect step ---
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save temp file
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.filename).suffix)
        f.save(tmp.name)
        input_path = tmp.name
    except Exception as e:
        return jsonify({'error': f'File save failed: {e}'}), 400

    # Run pipeline for preview (load/validate/schema/profile)
    cols = []
    sample_rows = 2000
    suffix = Path(f.filename).suffix.lower()
    try:
        # Use pipeline to get a full profile and inferred mapping for preview
        if HAS_PIPELINE:
            try:
                pcfg = PipelineConfig()
                pr = run_pipeline(input_path, Path(f.filename).name, mode='universal', cfg=pcfg)
                cols = list(pr.column_profile.get('numeric', []) + pr.column_profile.get('categorical', []) + pr.column_profile.get('text', []) + pr.column_profile.get('datetime', []) )
                profile = pr.meta
                inferred_mapping = pr.mapping
                warnings_list = pr.warnings or []
                session['preview_run_dir'] = pr.artifacts.get('output_dir')
                session['preview_meta'] = pr.meta
                session['preview_mapping'] = pr.mapping
                session['preview_columns'] = cols
            except Exception:
                cols = []
        else:
            cols = []
    except Exception:
        cols = []

    # Suggest targets by common names
    suggestions = [c for c in cols if c.lower() in ('target', 'label', 'price', 'y', 'value', 'realized', 'reward')]
    # Build configuration page. Tasks and modes are populated client-side
    # by calling /capabilities so the UI reflects the engine capabilities.
    target_options = '\n'.join([f'<option value="{c}"{" selected" if c in suggestions else ""}>{c}</option>' for c in cols])

    # Build mapping override controls
    mapping_keys = ['datetime','symbol','price','open','high','low','close','volume','pnl','equity','patient_id','billed_amount','paid_amount','doctor','procedure','status','payment_mode','strategy','side','qty','fees','slippage']
    mapping_controls = ''
    for mk in mapping_keys:
        options = '<option value="">-- none --</option>'
        options += '\n'.join([f'<option value="{c}">{c}</option>' for c in cols])
        mapping_controls += f'<label for="map_{mk}">{mk}:</label><select id="map_{mk}" name="mapping_{mk}">{options}</select><br>'

    # show profile summary if available
    profile_html = ''
    if session.get('preview_meta'):
        pm = session.get('preview_meta')
        profile_html = f"<p>Preview: {pm.get('rows','?')} rows × {pm.get('cols','?')} cols</p>"

    cfg_html = f"""
    <!doctype html>
    <html>
    <head>
        <title>Configure Analysis</title>
        <script>
        async function loadCapabilities() {{
            try {{
                const r = await fetch('/capabilities');
                const j = await r.json();
                const sel = document.getElementById('tasks');
                sel.innerHTML = '';
                j.tasks_supported.forEach(t => {{
                    const opt = document.createElement('option');
                    opt.value = t; opt.text = t.replace(/_/g,' ').replace(/\\b\\w/g, c=>c.toUpperCase());
                    sel.appendChild(opt);
                }});
                const modeSel = document.getElementById('mode');
                if (modeSel) {{
                    ['universal','clinic','trading'].forEach(m => {{
                        const o = document.createElement('option'); o.value = m; o.text = m[0].toUpperCase()+m.slice(1);
                        modeSel.appendChild(o);
                    }});
                }}
            }} catch (e) {{ console.warn('capabilities fetch failed', e); }}
        }}
        window.addEventListener('DOMContentLoaded', loadCapabilities);
        </script>
    </head>
    <body>
        <h2>Configure analysis for: {Path(f.filename).name}</h2>
        <form method="post" action="/analyze">
            <input type="hidden" name="confirm" value="1">
            <input type="hidden" name="input_path" value="{input_path}">
            <label for="tasks">Select one or more tasks (hold Ctrl/Cmd):</label><br>
            <select id="tasks" name="tasks" multiple size="8"></select>
            <br><br>
            <label for="mode">Mode:</label><br>
            <select id="mode" name="mode"><option value="universal">Universal</option><option value="clinic">Clinic</option><option value="trading">Trading</option></select>
            <br><br>
            <label for="target">Target column (for classification/regression):</label><br>
            <select id="target" name="target">
                <option value="">-- none --</option>
                {target_options}
            </select>
            <br><br>
            <label for="text_col">Text column (for text/nlp tasks):</label><br>
            <input type="text" id="text_col" name="text_col" placeholder="e.g., description">
            <br><br>
            <h3>Override semantic mapping (optional)</h3>
            {mapping_controls}
            <br>
            <button type="submit">Run Analysis</button>
        </form>
        <p><a href="/">Start over</a></p>
    </body>
    </html>
    """
    return cfg_html


@app.route('/artifacts/<run_id>/<path:filename>')
def artifacts(run_id: str, filename: str):
    """Serve artifact files from outputs/<run_id>/"""
    base = Path(__file__).parent / 'outputs' / run_id
    if not base.exists():
        return jsonify({'error': 'run_id not found'}), 404
    
    # Prevent directory traversal
    safe_path = (base / filename).resolve()
    if not str(safe_path).startswith(str(base.resolve())):
        return jsonify({'error': 'forbidden'}), 403
    
    if not safe_path.exists():
        return jsonify({'error': 'file not found'}), 404
    
    return send_from_directory(str(base), filename)


@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large errors."""
    limit_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({'error': f'File too large. Maximum size: {limit_mb}MB'}), 413


@app.route('/runs/<run_id>')
def run_page(run_id: str):
    """Display results page with rendered markdown and metrics."""
    base = Path(__file__).parent / 'outputs' / run_id
    if not base.exists():
        return f'<h1>Run {run_id} not found</h1>', 404
    
    # Read report files
    report_json = {}
    markdown_html = ''
    
    json_file = base / 'report.json'
    if json_file.exists():
        try:
            report_json = json.loads(json_file.read_text(encoding='utf-8'))
        except Exception:
            pass
    
    md_file = base / 'report.md'
    if md_file.exists() and HAS_MARKDOWN:
        try:
            md_text = md_file.read_text(encoding='utf-8')
            markdown_html = markdown2.markdown(md_text)
        except Exception:
            markdown_html = '<p>Could not render markdown</p>'
    
    # Collect charts
    charts = []
    charts_dir = base / 'charts'
    # Prefer canonical chart list from report.json if present
    report_charts = report_json.get('charts') or []
    if report_charts:
        # use filenames from report['charts'] in order, avoid duplicates
        seen = set()
        for c in report_charts:
            fn = c.get('filename')
            if not fn:
                continue
            if fn in seen:
                continue
            seen.add(fn)
            charts.append(fn)
    else:
        if charts_dir.exists():
            # de-duplicate and sort
            charts = sorted({p.name for p in charts_dir.iterdir() if p.is_file()})
    # Build standardized meta structure from report.json (support both formats)
    meta = report_json.get('meta', {}) or {}
    # If descriptive results exist, derive canonical fields
    desc = report_json.get('results', {}).get('descriptive') or {}
    if desc:
        shape = desc.get('shape', {})
        cols = desc.get('columns', [])
        meta.setdefault('rows', shape.get('rows') if isinstance(shape, dict) else shape)
        meta.setdefault('cols', shape.get('cols') if isinstance(shape, dict) else (len(cols) if cols else '?'))
        meta.setdefault('columns', cols)
        meta.setdefault('numeric_columns', desc.get('numeric_columns', []))
        meta.setdefault('categorical_columns', desc.get('categorical_columns', []))

    if not meta.get('run_id'):
        meta['run_id'] = report_json.get('run_id', run_id)
    if not meta.get('input'):
        input_info = report_json.get('input', {})
        meta['input'] = input_info.get('path') or input_info.get('source') or meta.get('input')
    if not meta.get('created_at'):
        meta['created_at'] = report_json.get('created_at')
    if not meta.get('task_type'):
        knobs = report_json.get('knobs', {})
        if isinstance(knobs, dict):
            meta['task_type'] = next((k for k, v in knobs.items() if v), 'unknown')
    meta.setdefault('rows', meta.get('rows', '?'))
    meta.setdefault('cols', meta.get('cols', '?'))

    results = report_json.get('results', {})
    metrics_html = '<div class="metrics">'
    if isinstance(results, dict):
        for task_name, task_result in results.items():
            if isinstance(task_result, dict):
                metrics = task_result.get('metrics', task_result)
                if 'accuracy' in metrics:
                    acc = metrics.get('accuracy', 0)
                    metrics_html += f'<p><strong>{task_name}:</strong> accuracy = {acc:.3f}</p>'
                elif 'r2' in metrics:
                    r2 = metrics.get('r2', 0)
                    rmse = metrics.get('rmse', 0)
                    metrics_html += f'<p><strong>{task_name}:</strong> R² = {r2:.3f}, RMSE = {rmse:.3f}</p>'
                elif 'mae' in metrics and 'rmse' in metrics:
                    mae = metrics.get('mae', 0)
                    rmse = metrics.get('rmse', 0)
                    metrics_html += f'<p><strong>{task_name}:</strong> MAE = {mae:.3f}, RMSE = {rmse:.3f}</p>'
                elif 'top_anomalies' in metrics or 'anomalies' in metrics:
                    table = metrics.get('top_anomalies') or metrics.get('anomalies')
                    if isinstance(table, list) and table:
                        metrics_html += f'<div><strong>{task_name} - Top anomalies</strong><table border="1" cellpadding="6" style="border-collapse:collapse;margin-top:8px;">'
                        cols_hdr = list(table[0].keys())
                        metrics_html += '<tr>' + ''.join(f'<th>{h}</th>' for h in cols_hdr) + '</tr>'
                        for r in table[:20]:
                            metrics_html += '<tr>' + ''.join(f'<td>{r.get(h, "")}</td>' for h in cols_hdr) + '</tr>'
                        metrics_html += '</table></div>'
    metrics_html += '</div>'
    try:
        trading = report_json.get('results', {}).get('trading_kpis')
        if isinstance(trading, dict):
            if 'total_pnl' in trading or 'win_rate' in trading or trading.get('by_symbol'):
                metrics_html += '<div class="metrics"><h3>Trading KPIs</h3>'
                tp = trading.get('total_pnl')
                if tp is not None:
                    metrics_html += f'<p><strong>Total PnL:</strong> {tp:.2f}</p>'
                wr = trading.get('win_rate')
                if wr is not None:
                    metrics_html += f'<p><strong>Win rate:</strong> {wr*100:.2f}%</p>'
                aw = trading.get('avg_win')
                if aw is not None:
                    metrics_html += f'<p><strong>Avg win:</strong> {aw:.2f}</p>'
                al = trading.get('avg_loss')
                if al is not None:
                    metrics_html += f'<p><strong>Avg loss:</strong> {al:.2f}</p>'
                bysym = trading.get('by_symbol') or {}
                if bysym:
                    metrics_html += '<p><strong>By symbol (top):</strong></p><ul>'
                    for s, v in list(bysym.items())[:10]:
                        metrics_html += f"<li>{s}: count={v.get('count',0)}, sum={v.get('sum',0):.2f}</li>"
                    metrics_html += '</ul>'
                charts_list = report_json.get('artifacts', {}).get('charts', [])
                eq_name = next((c for c in charts_list if 'equity' in c.lower()), None)
                if eq_name:
                    metrics_html += f'<p><a href="/artifacts/{run_id}/charts/{eq_name}">View equity curve</a></p>'
                metrics_html += '</div>'
    except Exception:
        pass

    try:
        kpis = report_json.get('kpis') or {}
        if isinstance(kpis, dict) and kpis.get('summary'):
            metrics_html += '<div class="metrics"><h3>KPIs</h3>'
            metrics_html += f"<p>{kpis.get('summary')}</p>"
            comp = kpis.get('computed', {})
            if isinstance(comp, dict) and comp:
                metrics_html += '<ul>'
                for k, v in list(comp.items())[:10]:
                    metrics_html += f'<li><strong>{k}:</strong> {v}</li>'
                metrics_html += '</ul>'
            tables = kpis.get('tables', {})
            for tname, tbl in (tables or {}).items():
                if isinstance(tbl, list) and tbl:
                    metrics_html += f'<div><strong>{tname}</strong><table border="1" cellpadding="6" style="border-collapse:collapse;margin-top:8px;">'
                    hdrs = list(tbl[0].keys())
                    metrics_html += '<tr>' + ''.join(f'<th>{h}</th>' for h in hdrs) + '</tr>'
                    for r in tbl[:10]:
                        metrics_html += '<tr>' + ''.join(f'<td>{r.get(h, "")}</td>' for h in hdrs) + '</tr>'
                    metrics_html += '</table></div>'
            metrics_html += '</div>'
    except Exception:
        pass

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Results - {run_id}</title>
        <style>
            * {{ font-family: 'Segoe UI', sans-serif; }}
            body {{ max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
            .header {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            h1 {{ color: #2c3e50; margin: 0; }}
            .run-id {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
            .meta {{ margin-top: 15px; padding: 10px; background: #ecf0f1; border-radius: 4px; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .metrics {{ background: #e8f8f5; padding: 15px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #27ae60; }}
            .metrics p {{ margin: 8px 0; }}
            .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }}
            .chart-item {{ background: #f9f9f9; padding: 15px; border-radius: 4px; text-align: center; }}
            .chart-item img {{ max-width: 100%; border-radius: 4px; }}
            .report {{ margin-top: 30px; line-height: 1.6; }}
            .report h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .report h3 {{ color: #7f8c8d; margin-top: 20px; }}
            .report code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
            .report pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 4px; overflow-x: auto; }}
            .links {{ margin-top: 20px; }}
            .links a {{ display: inline-block; margin-right: 15px; padding: 8px 15px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; }}
            .links a:hover {{ background: #2980b9; }}
            a.back {{ display: inline-block; margin-bottom: 20px; color: #3498db; text-decoration: none; }}
            a.back:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <a class="back" href="/">← Back to upload</a>
        
        <div class="header">
            <h1>📊 Analysis Complete</h1>
            <div class="run-id">Run ID: <code>{run_id}</code></div>
            <div class="meta">
                <strong>Data:</strong> {meta.get('rows', '?')} rows × {meta.get('cols', '?')} columns
                | <strong>Task:</strong> {meta.get('task_type', 'unknown')}
                | <strong>Input:</strong> {Path(meta.get('input', 'unknown')).name if meta.get('input') else 'unknown'}
            </div>
        </div>
        
        {metrics_html}
        
        {'<div class="container"><h2>📈 Charts</h2><div class="charts">' + ''.join(
            f'<div class="chart-item"><strong>{c}</strong><br><img src="/artifacts/{run_id}/charts/{c}" alt="{c}"></div>'
            for c in charts
        ) + '</div></div>' if charts else ''}
        
        <div class="container report">
            <h2>📋 Full Report</h2>
            {markdown_html if markdown_html else '<p>No markdown report available</p>'}
        </div>
        
        <div class="container links">
            <strong>Download:</strong>
            <a href="/artifacts/{run_id}/report.json">📄 report.json</a>
            <a href="/artifacts/{run_id}/report.md">📝 report.md</a>
        </div>
    </body>
    </html>
    """
    return html


@app.route('/capabilities')
def capabilities():
    """Return JSON describing engine capabilities for dynamic UI generation."""
    tasks = list(_get_supported_tasks().keys())
    models = []
    file_types = ['csv', 'excel', 'json', 'jsonl', 'txt', 'pdf', 'parquet']
    if BIG_ENGINE_AVAILABLE and BIG_ENGINE_MODULE:
        try:
            cfg_models = getattr(BIG_ENGINE_MODULE, 'DEFAULT_CONFIG', {}).get('models', {})
            models = list(cfg_models.keys())
        except Exception:
            models = []
    else:
        models = ['classification_model', 'regression_model', 'kmeans_k']

    return jsonify({'tasks_supported': tasks, 'models_supported': models, 'file_types_supported': file_types, 'multi_task_supported': bool(BIG_ENGINE_AVAILABLE)})


@app.route('/runs')
def runs_list():
    base = Path(__file__).parent / 'outputs'
    runs = []
    if base.exists():
        for d in sorted([p for p in base.iterdir() if p.is_dir()], reverse=True):
            try:
                rpt = d / 'report.json'
                if rpt.exists():
                    r = json.loads(rpt.read_text(encoding='utf-8'))
                    runs.append({
                        'run_id': r.get('run_id') or d.name,
                        'created_at': r.get('created_at'),
                        'input': r.get('input'),
                        'mode': r.get('mode'),
                        'rows': r.get('meta', {}).get('rows'),
                        'cols': r.get('meta', {}).get('cols'),
                        'summary': (r.get('kpis') or {}).get('summary','')
                    })
            except Exception:
                continue
    html = '<h1>Run History</h1><ul>'
    for r in runs:
        html += f"<li>{r.get('created_at')} — {r.get('run_id')} — {r.get('input')} — mode={r.get('mode')} — <a href='/runs/{r.get('run_id')}'>View</a> — <a href=\"/runs/delete/{r.get('run_id')}\" onclick=\"return confirm('Delete run?')\">Delete</a></li>"
    html += '</ul><p><a href="/">Back</a></p>'
    return html


@app.route('/runs/delete/<run_id>')
def runs_delete(run_id: str):
    base = Path(__file__).parent / 'outputs' / run_id
    if not base.exists():
        return redirect(url_for('runs_list'))
    # safety: only delete inside outputs
    try:
        import shutil
        shutil.rmtree(base)
    except Exception:
        pass
    return redirect(url_for('runs_list'))


@app.route('/compare', methods=['GET','POST'])
def compare_runs():
    base = Path(__file__).parent / 'outputs'
    runs = [p.name for p in sorted(base.iterdir()) if p.is_dir()]
    if request.method == 'POST':
        a = request.form.get('run_a')
        b = request.form.get('run_b')
        if not a or not b or a==b:
            return '<p>Select two different runs</p><p><a href="/compare">Back</a></p>'
        try:
            ra = json.loads((base / a / 'report.json').read_text())
            rb = json.loads((base / b / 'report.json').read_text())
        except Exception:
            return '<p>Could not load runs</p>'
        # basic comparison
        mode = ra.get('mode')
        comp_html = f'<h2>Comparing {a} vs {b} (mode={mode})</h2>'
        def val(r,k): return (r.get('kpis') or {}).get('computed',{}).get(k)
        if mode == 'clinic':
            keys = ['total_revenue_collected','total_visits','no_show_rate']
        elif mode == 'trading':
            keys = ['total_pnl','win_rate','max_drawdown']
        else:
            keys = ['rows','cols']
        comp_html += '<table border="1"><tr><th>metric</th><th>'+a+'</th><th>'+b+'</th></tr>'
        for k in keys:
            comp_html += f'<tr><td>{k}</td><td>{val(ra,k)}</td><td>{val(rb,k)}</td></tr>'
        comp_html += '</table><p><a href="/compare">Back</a></p>'
        return comp_html
    # GET
    html = '<h1>Compare Runs</h1><form method="post"><label>Run A:</label><select name="run_a">' + '\n'.join([f'<option value="{r}">{r}</option>' for r in runs]) + '</select>'
    html += '<label>Run B:</label><select name="run_b">' + '\n'.join([f'<option value="{r}">{r}</option>' for r in runs]) + '</select>'
    html += '<button type="submit">Compare</button></form><p><a href="/">Back</a></p>'
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
