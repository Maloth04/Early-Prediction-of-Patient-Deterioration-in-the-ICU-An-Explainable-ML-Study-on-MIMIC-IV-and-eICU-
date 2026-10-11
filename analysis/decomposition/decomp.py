#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One decomposition run to replace the three partly-inconsistent ones.

Everything below is computed on identical instances, identical splits and an
identical model configuration, so the block values, the paired differences and
the per-unit values all come from the same experiment. Differences between the
two feature blocks are taken WITHIN each split and their sign counted, which is
the design Section 4.7 established and which the earlier decomposition did not use.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
NSPLIT = 5
BEH = ('_missing', '_time_since_obs_h')
OUT = {}

icu = pd.read_csv(B + 'icustays_CLEANED.csv',
                  usecols=['stay_id', 'hadm_id', 'intime', 'outtime', 'first_careunit'],
                  low_memory=False)
adm = pd.read_csv('/mnt/user-data/uploads/admissions.csv',
                  usecols=['hadm_id', 'deathtime'], low_memory=False)
icu = icu.merge(adm, on='hadm_id', how='left')
for c in ('intime', 'outtime', 'deathtime'):
    icu[c] = pd.to_datetime(icu[c], errors='coerce')


def fit_eval(Xtr, ytr, Xte, yte, seed):
    pw = (len(ytr) - ytr.sum()) / ytr.sum()
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=seed)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    del m; gc.collect()
    return p, roc_auc_score(yte, p), average_precision_score(yte, p)


for H, HRS in (('4h', 4), ('8h', 8)):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    beh = [c for c in feats if c.endswith(BEH)]
    phy = [c for c in feats if c not in beh]
    assert len(feats) == 81 and len(phy) == 53 and len(beh) == 28, (len(feats), len(phy), len(beh))

    # corrected endpoint: escalation OR death inside the same horizon
    d = df.merge(icu[['stay_id', 'intime', 'outtime', 'deathtime', 'first_careunit']],
                 on='stay_id', how='left')
    d['k'] = d.groupby('stay_id').cumcount()
    lm = d['intime'] + pd.to_timedelta(24 + d['k'] * HRS, unit='h')
    hz = lm + pd.to_timedelta(HRS, unit='h')
    died = d.deathtime.notna() & (d.deathtime > lm) & (d.deathtime <= hz)
    y_spec = df['label'].values
    y_corr = ((y_spec == 1) | died.values).astype(int)
    unit = d['first_careunit'].values
    g = df['subject_id'].values
    print(f'[{H}] instances {len(df):,}  specified events {y_spec.sum():,}  '
          f'corrected events {y_corr.sum():,}  deaths added {int(y_corr.sum()-y_spec.sum()):,}',
          flush=True)

    res = {k: {b: {'auroc': [], 'auprc': []} for b in ('full', 'phys', 'beh')}
           for k in ('specified', 'corrected')}
    units = {}
    for s in range(NSPLIT):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s)
                      .split(df, y_spec, g))
        for key, y in (('specified', y_spec), ('corrected', y_corr)):
            for bname, cols in (('full', feats), ('phys', phy), ('beh', beh)):
                p, a, ap = fit_eval(df.iloc[tr][cols], y[tr], df.iloc[te][cols], y[te], 100 + s)
                res[key][bname]['auroc'].append(float(a))
                res[key][bname]['auprc'].append(float(ap))
                if key == 'specified' and bname == 'full':
                    for u in pd.unique(unit[te]):
                        if not isinstance(u, str):
                            continue
                        m = unit[te] == u
                        if y[te][m].sum() >= 5 and len(np.unique(y[te][m])) > 1:
                            units.setdefault(u, {'auroc': [], 'n': [], 'ev': []})
                            units[u]['auroc'].append(float(roc_auc_score(y[te][m], p[m])))
                            units[u]['n'].append(int(m.sum()))
                            units[u]['ev'].append(int(y[te][m].sum()))
        print(f'  [{H}] split {s+1}/{NSPLIT} done', flush=True)

    summ = {}
    for key in ('specified', 'corrected'):
        e = {}
        for b in ('full', 'phys', 'beh'):
            e[b] = dict(auroc_mean=float(np.mean(res[key][b]['auroc'])),
                        auroc_sd=float(np.std(res[key][b]['auroc'])),
                        auprc_mean=float(np.mean(res[key][b]['auprc'])),
                        auprc_sd=float(np.std(res[key][b]['auprc'])))
        d_bp = np.array(res[key]['beh']['auroc']) - np.array(res[key]['phys']['auroc'])
        e['beh_minus_phys'] = dict(mean=float(d_bp.mean()), sd=float(d_bp.std()),
                                   per_split=[round(x, 4) for x in d_bp],
                                   splits_positive=int((d_bp > 0).sum()), n=len(d_bp))
        bs = (np.array(res[key]['beh']['auroc']) - .5) / (np.array(res[key]['full']['auroc']) - .5)
        e['behaviour_share_of_above_chance'] = dict(mean=float(bs.mean()), sd=float(bs.std()))
        summ[key] = e
    summ['per_unit'] = {u: dict(auroc_mean=float(np.mean(v['auroc'])),
                                auroc_sd=float(np.std(v['auroc'])),
                                splits=len(v['auroc']),
                                median_instances=int(np.median(v['n'])),
                                median_events=int(np.median(v['ev'])))
                        for u, v in units.items()}
    OUT[H] = summ
    print(json.dumps(summ, indent=2)[:1400], flush=True)
    del df, d; gc.collect()

json.dump(OUT, open('/home/claude/rerun/decomp.json', 'w'), indent=2)
print('WROTE decomp.json')
