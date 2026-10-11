#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The check the review calls the principal outstanding threat: if arterial lines and
lactates are steps within escalation rather than observations before it, then excluding
events that occur inside the first hour after the landmark should cost the
measurement-behaviour block more than it costs physiology."""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
COLS = ['vent_starttime', 'arrest_starttime', 'vaso_starttime']
icu = pd.read_csv(B + 'icustays_CLEANED.csv', usecols=['stay_id', 'intime'], low_memory=False)
icu['intime'] = pd.to_datetime(icu.intime, errors='coerce')
ec = pd.read_csv(B + 'endpoint_components_CLEANED.csv', low_memory=False)
for c in COLS:
    ec[c] = pd.to_datetime(ec[c], errors='coerce')
OUT = {}

for H, HRS in (('4h', 4), ('8h', 8)):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    beh = [c for c in feats if c.endswith(('_missing', '_time_since_obs_h'))]
    phy = [c for c in feats if c not in beh]
    y = df['label'].values; g = df['subject_id'].values
    k = df.groupby('stay_id').cumcount().values
    m = df[['stay_id']].merge(icu, on='stay_id', how='left').merge(
        ec[['stay_id'] + COLS], on='stay_id', how='left')
    lm = m['intime'] + pd.to_timedelta(24 + k * HRS, unit='h')
    lead = (m[COLS].min(axis=1) - lm).dt.total_seconds().values / 3600

    for LAG in (1.0, 2.0):
        drop = (y == 1) & (lead >= 0) & (lead < LAG)
        keep = ~drop
        res = {b: [] for b in ('full', 'phys', 'beh')}
        for s in range(5):
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s)
                          .split(df, y, g))
            tr, te = tr[keep[tr]], te[keep[te]]
            pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
            for bn, cols in (('full', feats), ('phys', phy), ('beh', beh)):
                mo = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=100 + s)
                mo.fit(df.iloc[tr][cols], y[tr])
                res[bn].append(float(roc_auc_score(y[te], mo.predict_proba(df.iloc[te][cols])[:, 1])))
                del mo; gc.collect()
            print(f'  [{H}] lag {LAG}h split {s+1}/5', flush=True)
        f_ = np.array(res['full']); bh = np.array(res['beh']); ph = np.array(res['phys'])
        OUT.setdefault(H, {})[f'lag_{LAG:g}h'] = dict(
            events_dropped=int(drop.sum()), events_remaining=int((y == 1).sum() - drop.sum()),
            pct_dropped=round(100 * drop.sum() / (y == 1).sum(), 1),
            full=float(f_.mean()), full_sd=float(f_.std()),
            phys=float(ph.mean()), phys_sd=float(ph.std()),
            beh=float(bh.mean()), beh_sd=float(bh.std()),
            behaviour_share=float(np.mean((bh - .5) / (f_ - .5))),
            physiology_share=float(np.mean((ph - .5) / (f_ - .5))))
        print(H, LAG, OUT[H][f'lag_{LAG:g}h'], flush=True)
    del df; gc.collect()

json.dump(OUT, open('/home/claude/rev3/blank.json', 'w'), indent=2)
print('WROTE blank.json')
