#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three analyses the sharpened design specified and the thesis never carried:
what the behaviour block is made of, which clinical decisions it records, and
what the model costs a unit in alerts."""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
OUT = {}

for H in ('4h', '8h'):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    miss = [c for c in feats if c.endswith('_missing')]
    rec = [c for c in feats if c.endswith('_time_since_obs_h')]
    y = df['label'].values; g = df['subject_id'].values
    print(f'[{H}] {len(miss)} presence + {len(rec)} recency features', flush=True)

    # ---- sub-decomposition of the behaviour block, same five splits ----
    res = {k: [] for k in ('recency', 'presence', 'both')}
    for s in range(5):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s).split(df, y, g))
        for name, cols in (('recency', rec), ('presence', miss), ('both', miss + rec)):
            pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
            m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=100 + s)
            m.fit(df.iloc[tr][cols], y[tr])
            res[name].append(float(roc_auc_score(y[te], m.predict_proba(df.iloc[te][cols])[:, 1])))
            del m; gc.collect()
        print(f'  [{H}] sub-decomposition split {s+1}/5', flush=True)
    OUT.setdefault(H, {})['behaviour_sub'] = {
        k: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k, v in res.items()}

    # ---- which clinical decision each presence feature records ----
    if H == '4h':
        rows = []
        for c in miss:
            var = c[:-8]
            present = df[c] == 0
            n_p, n_a = int(present.sum()), int((~present).sum())
            if min(n_p, n_a) < 500:
                continue
            rp = 100 * y[present.values].mean()
            ra = 100 * y[(~present).values].mean()
            rows.append(dict(variable=var, pct_present=round(100 * n_p / len(df), 1),
                             rate_present=round(rp, 3), rate_absent=round(ra, 3),
                             ratio=round(rp / ra, 2) if ra else None))
        rows.sort(key=lambda r: -(r['ratio'] or 0))
        OUT[H]['presence_mechanism'] = rows

        # ---- the truncation objection ----
        hr = 'hr_time_since_obs_h'
        if hr in df.columns:
            ev, ne = df.loc[y == 1, hr], df.loc[y == 0, hr]
            OUT[H]['recency_distribution'] = dict(
                event_median=float(ev.median()), nonevent_median=float(ne.median()),
                event_mean=float(ev.mean()), nonevent_mean=float(ne.mean()),
                event_p90=float(ev.quantile(.9)), nonevent_p90=float(ne.quantile(.9)))

        # ---- alarm burden at the point of use ----
        tr, rest = next(GroupShuffleSplit(1, test_size=0.4, random_state=100).split(df, y, g))
        gr = df['subject_id'].values[rest]
        cal, te = next(GroupShuffleSplit(1, test_size=0.5, random_state=7)
                       .split(df.iloc[rest], y[rest], gr))
        cal, te = rest[cal], rest[te]
        pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
        m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=100)
        m.fit(df.iloc[tr][feats], y[tr])
        pc = m.predict_proba(df.iloc[cal][feats])[:, 1]
        pt = m.predict_proba(df.iloc[te][feats])[:, 1]
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        platt = LogisticRegression(max_iter=1000).fit(lg(pc).reshape(-1, 1), y[cal])
        ptr = platt.predict_proba(lg(pt).reshape(-1, 1))[:, 1]
        yt = y[te]
        band = []
        for sens in (0.9, 0.8, 0.7, 0.6, 0.5):
            thr = np.quantile(ptr[yt == 1], 1 - sens)
            fire = ptr >= thr
            alerts = int(fire.sum()); tp = int(yt[fire].sum())
            ppv = tp / alerts if alerts else 0
            per1000 = 1000 * alerts / len(yt)
            band.append(dict(sensitivity=int(sens * 100),
                             alerts_per_1000_instances=round(per1000, 1),
                             ppv_pct=round(100 * ppv, 2),
                             number_needed_to_alert=int(round(1 / ppv)) if ppv else None,
                             alerts_per_bed_day=round(6 * per1000 / 1000, 2),
                             alerts_20_bed_unit_per_day=int(round(20 * 6 * per1000 / 1000))))
        OUT[H]['alarm_burden'] = band
        OUT[H]['burden_test_instances'] = int(len(yt))
        OUT[H]['burden_test_events'] = int(yt.sum())
        del m; gc.collect()
    del df; gc.collect()

json.dump(OUT, open('/home/claude/rerun/mech.json', 'w'), indent=2)
print(json.dumps(OUT, indent=2)[:3000])
