from .base import KPIEngineBase, KPIResult
from .utils import resolve_mapping, safe_float
from .formatting import currency_fmt
import pandas as pd
import numpy as np
from typing import Optional

class ClinicKPIEngine(KPIEngineBase):
    mode_name = 'clinic'

    def compute(self, pr, config=None):
        cfg = config or self.cfg
        res = KPIResult(mode='clinic')
        df = pr.df_clean
        schema = pr.column_profile or {}
        mapping = resolve_mapping(pr.mapping or {}, schema)

        # use pipeline role inference if present
        roles = {}
        dataset_type = None
        if isinstance(pr.column_profile, dict):
            roles = pr.column_profile.get('roles', {}) or {}
            dataset_type = pr.column_profile.get('dataset_type')

        # targeted processing for lab results, vitals, visits, billing
        if dataset_type == 'lab_results_table':
            try:
                test_col = roles.get('test_name') or col('test_name') or 'test_name'
                value_col = roles.get('test_value') or col('test_value') or 'value'
                unit_col = roles.get('unit') or col('unit')
                pid = roles.get('patient_id') or col('patient_id')
                # basic lab KPIs
                res.computed['total_tests'] = int(len(df))
                if pid and pid in df.columns:
                    res.computed['unique_patients'] = int(df[pid].nunique())
                # top tests
                try:
                    top = df[test_col].value_counts().head(20).to_dict()
                    res.tables['top_lab_tests'] = [{'test':k,'count':int(v)} for k,v in top.items()]
                    res.charts.append({'id':'clinic_top_lab_tests','chart_type':'bar','title':'Top Lab Tests','data': res.tables['top_lab_tests'],'x':'test','y':'count','priority':3})
                except Exception:
                    pass
                # abnormal percent if flag exists
                if roles.get('abnormal_flag') and roles.get('abnormal_flag') in df.columns:
                    try:
                        af = df[roles.get('abnormal_flag')].astype(str).str.lower()
                        abnormal = af.isin(['1','true','yes','y','abnormal'])
                        res.computed['abnormal_percent'] = float(abnormal.sum())/max(1,len(df))
                    except Exception:
                        pass
                res.summary = 'Lab results table with top tests and abnormal rates where available.'
                return res
            except Exception:
                res.warnings.append('lab_results_processing_failed')

        if dataset_type == 'vitals_table':
            try:
                # detect vitals columns and trend them
                vitals = ['bp','blood_pressure','systolic','diastolic','hr','heart_rate','spo2','weight']
                found = [c for c in df.columns if any(v in str(c).lower() for v in vitals)]
                res.computed['vitals_columns_detected'] = len(found)
                # basic trend charts for numeric vitals
                for c in found:
                    try:
                        ser = pd.to_numeric(df[c], errors='coerce')
                        res.tables.setdefault('vitals_trends', []).append({'vital': c, 'summary': ser.describe().to_dict()})
                        res.charts.append({'id': f'vital_trend_{c}','chart_type':'line','title':f'{c} over time','data':[{'index':i,'value':float(v)} for i,v in enumerate(ser.fillna(method='ffill').tolist())],'x':'index','y':'value','priority':3})
                    except Exception:
                        continue
                res.summary = 'Vitals table with trend lines for detected vital signs.'
                return res
            except Exception:
                res.warnings.append('vitals_processing_failed')

        if dataset_type == 'appointments_or_visits_table':
            try:
                # visits per day/week/month
                dtc = roles.get('datetime') or col('datetime')
                if dtc and dtc in df.columns:
                    ser = pd.to_datetime(df[dtc], errors='coerce')
                    df['_kpi_dt'] = ser.dt.date
                    per_day = df.groupby('_kpi_dt').size().reset_index()
                    res.tables['visits_per_day'] = per_day.head(200).to_dict('records')
                    res.charts.append({'id':'clinic_visits_per_day','chart_type':'time_series','title':'Visits Per Day','x':'_kpi_dt','y':'count','priority':2})
                res.computed['total_visits'] = int(len(df))
                res.summary = 'Visits/appointments table: trends by day/week/month.'
                return res
            except Exception:
                res.warnings.append('visits_processing_failed')

        if dataset_type == 'billing_or_revenue_table':
            try:
                paid_col = roles.get('billing_amount') or roles.get('paid_amount') or col('paid_amount') or col('paid')
                billed_col = roles.get('billing_amount') or roles.get('billed_amount') or col('billed_amount') or col('billed')
                dtc = roles.get('datetime') or col('datetime')
                if paid_col and paid_col in df.columns:
                    res.computed['total_revenue_collected'] = float(pd.to_numeric(df[paid_col], errors='coerce').fillna(0).sum())
                if billed_col and billed_col in df.columns:
                    res.computed['total_revenue_billed'] = float(pd.to_numeric(df[billed_col], errors='coerce').fillna(0).sum())
                if dtc and dtc in df.columns:
                    ser = pd.to_datetime(df[dtc], errors='coerce')
                    df['_kpi_dt_date'] = ser.dt.date
                    daily = df.groupby('_kpi_dt_date')[paid_col].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()).reset_index()
                    res.tables['revenue_by_day'] = daily.head(200).to_dict('records')
                    res.charts.append({'id': 'clinic_revenue_by_day','chart_type':'time_series','title':'Revenue by Day','x':'_kpi_dt_date','y': paid_col,'priority': 2,})
                res.summary = 'Billing table: revenue totals and daily breakdowns where available.'
                return res
            except Exception:
                res.warnings.append('billing_processing_failed')

        if df is None:
            res.warnings.append('No table data for clinic KPIs')
            return res

        # helper to get column by semantic key
        def col(k):
            return mapping.get(k)

        # visits
        res.computed['total_visits'] = int(len(df))

        # revenue: prefer paid_amount then billed_amount
        paid_col = col('paid_amount')
        billed_col = col('billed_amount')

        paid_sum = None
        billed_sum = None
        if paid_col and paid_col in df.columns:
            paid_sum = float(pd.to_numeric(df[paid_col], errors='coerce').fillna(0).sum())
        if billed_col and billed_col in df.columns:
            billed_sum = float(pd.to_numeric(df[billed_col], errors='coerce').fillna(0).sum())

        if paid_sum is not None:
            res.computed['total_revenue_collected'] = paid_sum
            res.used_columns['paid_amount'] = paid_col
        if billed_sum is not None:
            res.computed['total_revenue_billed'] = billed_sum
            res.used_columns['billed_amount'] = billed_col

        # avg revenue per visit
        try:
            if paid_sum is not None:
                res.computed['avg_revenue_per_visit'] = paid_sum / max(1, len(df))
            elif billed_sum is not None:
                res.computed['avg_revenue_per_visit'] = billed_sum / max(1, len(df))
        except Exception:
            pass

        # revenue trend if datetime exists
        dt_col = col('datetime')
        if dt_col and dt_col in df.columns:
            try:
                ser = pd.to_datetime(df[dt_col], errors='coerce')
                df['_kpi_dt_date'] = ser.dt.date
                df['_kpi_dt_week'] = ser.dt.to_period('W').apply(lambda p: p.start_time.date() if pd.notnull(p) else None)
                df['_kpi_dt_month'] = ser.dt.to_period('M').apply(lambda p: p.start_time.date() if pd.notnull(p) else None)
                # revenue by day/week/month
                if paid_col and paid_col in df.columns:
                    try:
                        daily = df.groupby('_kpi_dt_date')[paid_col].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()).reset_index()
                        res.tables['revenue_by_day'] = daily.head(200).to_dict('records')
                        res.charts.append({'id': 'clinic_revenue_by_day','chart_type':'time_series','title':'Revenue by Day','x':'_kpi_dt_date','y': paid_col,'priority': 2,})
                    except Exception:
                        res.warnings.append('Could not compute revenue by day')
                if paid_col and paid_col in df.columns:
                    try:
                        weekly = df.groupby('_kpi_dt_week')[paid_col].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()).reset_index()
                        res.tables['revenue_by_week'] = weekly.head(200).to_dict('records')
                        res.charts.append({'id': 'clinic_revenue_by_week','chart_type':'time_series','title':'Revenue by Week','x':'_kpi_dt_week','y': paid_col,'priority': 2,})
                    except Exception:
                        res.warnings.append('Could not compute revenue by week')
                if paid_col and paid_col in df.columns:
                    try:
                        monthly = df.groupby('_kpi_dt_month')[paid_col].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum()).reset_index()
                        res.tables['revenue_by_month'] = monthly.head(200).to_dict('records')
                        res.charts.append({'id': 'clinic_revenue_by_month','chart_type':'time_series','title':'Revenue by Month','x':'_kpi_dt_month','y': paid_col,'priority': 2,})
                    except Exception:
                        res.warnings.append('Could not compute revenue by month')
            except Exception:
                res.warnings.append('Datetime parsing failed for revenue trends')

        # by doctor
        doc_col = col('doctor')
        if doc_col and doc_col in df.columns and (paid_col or billed_col):
            amt_col = paid_col or billed_col
            try:
                sums = df.groupby(doc_col)[amt_col].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum())
                counts = df.groupby(doc_col).size()
                merged = pd.DataFrame({'revenue': sums, 'visits': counts}).fillna(0)
                merged = merged.sort_values('revenue', ascending=False).head(50)
                res.tables['revenue_by_doctor'] = [{'doctor': idx, 'revenue': float(r['revenue']), 'visits': int(r['visits'])} for idx, r in merged.iterrows()]
                res.charts.append({'id': 'clinic_revenue_by_doctor','chart_type': 'bar','title': 'Revenue by Doctor','data': res.tables['revenue_by_doctor'],'priority': 3,})
            except Exception:
                res.warnings.append('Could not compute doctor revenue/visits')

        # procedures
        proc_col = col('procedure')
        if proc_col and proc_col in df.columns and (paid_col or billed_col):
            amt_col = paid_col or billed_col
            try:
                sums = df.groupby(proc_col)[amt_col].apply(lambda x: pd.to_numeric(x, errors='coerce').fillna(0).sum())
                counts = df[proc_col].value_counts()
                total_count = int(counts.sum()) if counts.size>0 else 0
                top_by_revenue = sums.sort_values(ascending=False).head(50)
                top_by_count = counts.head(50)
                res.tables['top_procedures_by_revenue'] = [{'procedure': k, 'revenue': float(v)} for k,v in top_by_revenue.items()]
                res.tables['top_procedures_by_count'] = [{'procedure': k, 'count': int(v), 'share': (int(v)/total_count if total_count>0 else None)} for k,v in top_by_count.items()]
                # procedure mix percentages for top procedures
                try:
                    mix = (counts / total_count).head(20).fillna(0)
                    res.tables['procedure_mix'] = [{'procedure': k, 'share': float(v)} for k,v in mix.items()]
                except Exception:
                    pass
                res.charts.append({'id': 'clinic_top_procedures','chart_type': 'bar','title': 'Top Procedures by Revenue','data': res.tables['top_procedures_by_revenue'],'priority': 3,})
            except Exception:
                res.warnings.append('Could not compute procedures KPIs')

        # status normalization and rates
        status_col = col('status')
        if status_col and status_col in df.columns:
            try:
                s = df[status_col].astype(str).str.lower().str.replace('[^a-z0-9_-]', ' ', regex=True)
                no_show_keys = set(['no show','no-show','noshow','no_show'])
                cancel_keys = set(['cancel','cancelled','canceled','cxl'])
                completed_keys = set(['completed','done','attended'])
                def classify(x):
                    x = x.strip()
                    if any(k in x for k in no_show_keys):
                        return 'no_show'
                    if any(k in x for k in cancel_keys):
                        return 'cancelled'
                    if any(k in x for k in completed_keys):
                        return 'completed'
                    return 'other'
                classes = s.fillna('other').apply(classify)
                counts = classes.value_counts()
                res.computed['cancellation_rate'] = float(counts.get('cancelled',0) / max(1, counts.sum()))
                res.computed['no_show_rate'] = float(counts.get('no_show',0) / max(1, counts.sum()))
                res.tables['status_counts'] = [{'status': k, 'count': int(v)} for k,v in counts.items()]
            except Exception:
                res.warnings.append('Could not normalize status column')

        # outstanding
        if billed_sum is not None and paid_sum is not None:
            try:
                res.computed['outstanding_amount'] = billed_sum - paid_sum
            except Exception:
                res.warnings.append('Could not compute outstanding amount')

        # repeat patient rate
        pid = col('patient_id')
        if pid and pid in df.columns:
            try:
                counts = df[pid].value_counts()
                repeat = counts[counts>1].sum()
                res.computed['repeat_patient_rate'] = float(repeat / max(1, len(df)))
                res.used_columns['patient_id'] = pid
            except Exception:
                res.warnings.append('Could not compute repeat patient rate')

        # summary
        parts = []
        parts.append(f"This dataset contains {res.computed.get('total_visits',0)} visits.")
        if res.computed.get('total_revenue_collected'):
            parts.append(f"Total collected revenue is {currency_fmt(res.computed.get('total_revenue_collected'), cfg.currency)}.")
        elif res.computed.get('total_revenue_billed'):
            parts.append(f"Total billed revenue is {currency_fmt(res.computed.get('total_revenue_billed'), cfg.currency)}.")
        if res.computed.get('outstanding_amount'):
            parts.append(f"Outstanding receivables are {currency_fmt(res.computed.get('outstanding_amount'), cfg.currency)}.")
        if 'no_show_rate' in res.computed:
            parts.append(f"No-show rate is {res.computed.get('no_show_rate')*100:.1f}%.")

        res.summary = ' '.join(parts)
        return res
