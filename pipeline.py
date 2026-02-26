from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import time
import uuid
import traceback

from .constants import DEFAULT_CONFIG
from .loader import load
from .validator import validate_table
from .schema import infer_schema, infer_column_roles, infer_dataset_type
from .cleaner import normalize_columns, drop_constant_columns, handle_missing, build_model_frame
from .profiler import profile
from .artifacts import make_run_dirs, write_report, save_csv
from kpis.registry import get_engine



@dataclass
class PipelineConfig:
    max_upload_mb: int = DEFAULT_CONFIG['max_upload_mb']
    max_rows: int = DEFAULT_CONFIG['max_rows']
    max_cols: int = DEFAULT_CONFIG['max_cols']
    max_onehot_categories: int = 50
    missing_threshold_drop: float = DEFAULT_CONFIG['missing_threshold_drop']
    enable_dedup: bool = DEFAULT_CONFIG['enable_dedup']
    enable_cleaned_csv_export: bool = DEFAULT_CONFIG['enable_cleaned_csv_export']
    enable_raw_preview_export: bool = DEFAULT_CONFIG['enable_raw_preview_export']
    default_datetime_timezone: str = DEFAULT_CONFIG['default_datetime_timezone']
    allow_ocr: bool = DEFAULT_CONFIG['allow_ocr']
    output_dir: str = "outputs"
    max_model_columns: int = 200


@dataclass
class PipelineResult:
    run_id: str
    input_name: str
    source_type: str
    kind: str
    df_raw: Optional[Any]
    df_clean: Optional[Any]
    df_model: Optional[Any]
    text: Optional[str]
    column_profile: Dict[str, Any]
    mapping: Dict[str, Any]
    warnings: list
    errors: list
    artifacts: Dict[str, Any]
    meta: Dict[str, Any]


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def run_pipeline(file_path: str, filename: str, mode: str = 'universal', cfg: Optional[PipelineConfig] = None, mapping_overrides: Optional[Dict[str,str]] = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    run_id = _now_id()
    start = time.time()
    warnings = []
    errors = []
    df_raw = None
    df_clean = None
    df_model = None
    text = None
    column_profile = {}
    mapping = {}

    # load
    try:
        loaded = load(file_path, { 'allow_ocr': cfg.allow_ocr })
        kind = loaded.get('kind')
        source_type = loaded.get('source_type')
        if kind == 'table':
            df_raw = loaded.get('df')
        else:
            text = loaded.get('text')
    except Exception as e:
        errors.append(str(e))
        return PipelineResult(run_id, filename, 'unknown', 'unknown', None, None, None, None, {}, {}, warnings, errors, {}, {'elapsed': time.time()-start})

    # validate
    if df_raw is not None:
        try:
            df_raw, vwarns, verrs = validate_table(df_raw, file_path, { 'max_upload_mb': cfg.max_upload_mb })
            warnings.extend(vwarns)
            errors.extend(verrs)
        except Exception as e:
            errors.append(str(e))

    # infer schema
    if df_raw is not None and not errors:
        sch = infer_schema(df_raw, { 'max_onehot_categories': cfg.max_onehot_categories })
        column_profile = sch.get('types', {})
        schema_mapping = sch.get('mapping', {})
        # infer semantic roles and dataset type and attach to column_profile
        try:
            roles = infer_column_roles(df_raw)
            dataset_type = infer_dataset_type(df_raw, roles)
            column_profile['roles'] = roles
            column_profile['dataset_type'] = dataset_type
        except Exception:
            column_profile.setdefault('roles', {})
            column_profile.setdefault('dataset_type', 'unknown')

        # build user mapping from overrides (preferred)
        user_mapping = {}
        if mapping_overrides:
            try:
                for k, v in (mapping_overrides or {}).items():
                    if v:
                        user_mapping[k] = v
            except Exception:
                pass

        # resolve mapping: canonical keys -> column names using schema types
        try:
            from kpis.utils import resolve_mapping as _resolve_mapping
            mapping = _resolve_mapping(user_mapping, column_profile)
        except Exception:
            mapping = user_mapping or {}

        # clean
        try:
            dfc, norm_log = normalize_columns(df_raw)
            dfc, consts = drop_constant_columns(dfc)
            dfc, miss_log = handle_missing(dfc, { 'missing_threshold_drop': cfg.missing_threshold_drop })
            df_clean = dfc
            warnings.extend(norm_log if norm_log else [])
        except Exception as e:
            errors.append(f"cleaning_failed: {e}")

        # build model frame
        try:
            df_model = build_model_frame(df_clean, sch, { 'missing_threshold_drop': cfg.missing_threshold_drop, 'max_model_columns': cfg.max_model_columns, 'max_onehot_categories': cfg.max_onehot_categories })
        except Exception as e:
            errors.append(f"build_model_failed: {e}")

        # profile
        try:
            prof = profile(df_clean)
        except Exception:
            prof = {}

    else:
        prof = {}
        sch = { 'types': {}, 'mapping': {} }

    # make artifacts
    outdirs = make_run_dirs(cfg.output_dir, run_id)
    artifacts = { 'output_dir': outdirs['output_dir'], 'charts_dir': outdirs['charts_dir'], 'files': [] }

    # save cleaned csv if enabled
    try:
        if cfg.enable_cleaned_csv_export and df_clean is not None:
            cleaned_path = Path(outdirs['output_dir']) / 'cleaned.csv'
            save_csv(df_clean, str(cleaned_path))
            artifacts['files'].append('cleaned.csv')
        if cfg.enable_raw_preview_export and df_raw is not None:
            preview = df_raw.head(min(len(df_raw), 1000))
            preview_path = Path(outdirs['output_dir']) / 'raw_preview.csv'
            save_csv(preview, str(preview_path))
            artifacts['files'].append('raw_preview.csv')
    except Exception:
        pass

    meta = {
        'run_id': run_id,
        'input_name': filename,
        'source_type': source_type if df_raw is not None else 'document',
        'kind': 'table' if df_raw is not None else 'document',
        'rows': int(df_raw.shape[0]) if df_raw is not None else 0,
        'cols': int(df_raw.shape[1]) if df_raw is not None else 0,
        'elapsed_seconds': time.time() - start,
    }

    report = {
        'run_id': run_id,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'mode': mode,
        'input': filename,
        'meta': meta,
        'profile': prof,
        'mapping': mapping,
        'warnings': warnings,
        'errors': errors,
    }
    # create PipelineResult object so KPI engines can inspect dfs
    pr = PipelineResult(
        run_id=run_id,
        input_name=filename,
        source_type=meta.get('source_type'),
        kind=meta.get('kind'),
        df_raw=df_raw,
        df_clean=df_clean,
        df_model=df_model,
        text=text,
        column_profile=column_profile,
        mapping=mapping,
        warnings=warnings,
        errors=errors,
        artifacts=artifacts,
        meta=meta,
    )

    # compute KPIs using registry
    try:
        engine = get_engine(mode)
        kpi_res = engine.compute(pr)
        # attach kpis to report
        report['kpis'] = {
            'mode': kpi_res.mode,
            'computed': kpi_res.computed,
            'tables': kpi_res.tables,
            'charts': kpi_res.charts,
            'summary': kpi_res.summary,
            'warnings': kpi_res.warnings,
            'used_columns': kpi_res.used_columns,
        }
        # append any kpi warnings
        if kpi_res.warnings:
            report.setdefault('warnings', []).extend(kpi_res.warnings)
    except Exception as e:
        tb = traceback.format_exc()
        report.setdefault('warnings', []).append(f'KPI engine failed: {e}')
        report.setdefault('warnings', []).append(tb)
        # Ensure 'kpis' key exists even if engine failed so callers/tests don't error
        report['kpis'] = {
            'mode': mode,
            'computed': {},
            'tables': {},
            'charts': [],
            'summary': '',
            'warnings': [str(e), tb],
            'used_columns': [],
        }

    # Render charts from KPI descriptors using charts renderer
    try:
        from charts.renderer import ChartRenderer
        renderer = ChartRenderer(outdirs['output_dir'], max_samples=2000)
        saved = renderer.render_all(pr, kpi_res, report.get('results', {}), cfg=cfg)
        # attach to report
        report['charts'] = saved
        # also add to artifacts listing
        for s in saved:
            if s.get('filename') and s.get('filename') not in artifacts['files']:
                artifacts['files'].append(s.get('filename'))
    except Exception as e:
        report.setdefault('warnings', []).append(f'Chart rendering failed: {e}')

    try:
        write_report(report, outdirs['output_dir'])
    except Exception:
        pass

    pr = PipelineResult(
        run_id=run_id,
        input_name=filename,
        source_type=meta.get('source_type'),
        kind=meta.get('kind'),
        df_raw=df_raw,
        df_clean=df_clean,
        df_model=df_model,
        text=text,
        column_profile=column_profile,
        mapping=mapping,
        warnings=warnings,
        errors=errors,
        artifacts=artifacts,
        meta=meta,
    )

    return pr
