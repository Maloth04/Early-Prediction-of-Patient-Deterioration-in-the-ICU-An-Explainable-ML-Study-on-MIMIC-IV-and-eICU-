#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clustered interval for the CENTRED union statistic.

Section 3.6 registers the H3 decision rule on the lower bound of the interval,
not on the point estimate, so reporting the centred 0.726 as 'clearing 0.7' on
its own applies the wrong rule. The interval is computed here on the same 500
patient-clustered resamples used for the uncentred statistic.

Centring is per cohort, so each resample is centred on its own mean: the
external attributions are |SHAP - mean(SHAP)| over the resampled instances, and
the internal reference is the centred internal value held fixed.
"""
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import shap

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
OUT = '/home/claude/extval/'
SEED, NBOOT = 42, 500
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
rng = np.random.default_rng(SEED)
res = {}


def union_rho(si, se, k=15):
    oi = list(np.argsort(-si)); oe = list(np.argsort(-se))
    u = sorted(set(oi[:k]) | set(oe[:k]))
    return float(spearmanr(si[u], se[u]).statistic), len(u)


for H in ('4h', '8h'):
    dev = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    ext = pd.read_csv(OUT + f'eicu_features_{H}.csv', low_memory=False)
    ALL = [c for c in dev.columns if c not in ('stay_id', 'subject_id', 'label')]
    F = [c for c in ALL if not c.startswith('temp_f_')]
    if 'sid' in ext.columns:
        ext = ext.rename(columns={'sid': 'stay_id'})
    if 'subject_id' not in ext.columns:
        ext['subject_id'] = ext['stay_id']

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
                  .split(dev, groups=dev.subject_id.values))
    ytr = dev.label.values[tr]
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                          random_state=SEED)
    m.fit(dev.iloc[tr][F], ytr)
    expl = shap.TreeExplainer(m)
    SVi = expl.shap_values(dev.iloc[te][F])
    SVe = expl.shap_values(ext[F])

    si_c = np.abs(SVi - SVi.mean(axis=0)).mean(axis=0)     # centred internal reference
    pt, nu = union_rho(si_c, np.abs(SVe - SVe.mean(axis=0)).mean(axis=0))

    gq = ext.subject_id.values; uq = np.unique(gq)
    idx = {q: np.flatnonzero(gq == q) for q in uq}
    boot = []
    for _ in range(NBOOT):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idx[q] for q in pick])
        S = SVe[ii]
        boot.append(union_rho(si_c, np.abs(S - S.mean(axis=0)).mean(axis=0))[0])
    boot = np.array(boot)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    frac = float((boot >= 0.7).mean())
    res[H] = {'centred_union_rho': round(pt, 4), 'union_size': nu,
              'ci': [round(lo, 4), round(hi, 4)], 'frac_above_0.7': round(frac, 4),
              'registered_rule_met': bool(lo > 0.7)}
    print(f'{H}: centred union rho {pt:.4f} (union of {nu}), '
          f'clustered 95% interval [{lo:.3f}, {hi:.3f}], '
          f'{frac:.1%} of resamples >= 0.7 -> registered rule (lower bound > 0.7): '
          f'{"MET" if lo > 0.7 else "NOT MET"}')

json.dump(res, open(OUT + 'h3_centred_ci.json', 'w'), indent=1)
print('written h3_centred_ci.json')
