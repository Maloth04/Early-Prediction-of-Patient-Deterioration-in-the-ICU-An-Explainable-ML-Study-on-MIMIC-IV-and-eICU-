#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent replication of the Section 4.12 figures.

Difference from my first run: the six Fahrenheit predictors are DROPPED from
both cohorts rather than synthesised from Celsius, which is the harmonisation
Section 4.12.1 records. Synthesising them makes temp_f_missing flip from 2 per
cent in MIMIC-IV to 91 per cent in eICU, which is a fabricated distribution
shift in a predictor the model relies on.

Intervals are patient-clustered bootstrap percentiles, to match the reported
method rather than the analytic approximation used in my first pass.
"""
import numpy as np, pandas as pd, json, gc
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
OUT = '/home/claude/extval/'
SEED = 42
NBOOT = 1000
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
BEH = ('_missing', '_time_since_obs_h')
rng = np.random.default_rng(SEED)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def clustered_boot(y, p, groups, stat, n=NBOOT):
    uq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uq}
    out = []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        try:
            out.append(stat(yy, p[ii]))
        except Exception:
            pass
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


def slope_intercept(y, p):
    lr = LogisticRegression(max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


dev = pd.read_csv(B + 'engineered_features_4h_FULLSCALE.csv', low_memory=False)
ext = pd.read_csv(OUT + 'eicu_features_4h.csv', low_memory=False)

ALL = [c for c in dev.columns if c not in ('stay_id', 'subject_id', 'label')]
DROP = [c for c in ALL if c.startswith('temp_f_')]
FEATS = [c for c in ALL if c not in DROP]
PHYS = [c for c in FEATS if not c.endswith(BEH)]
BEHF = [c for c in FEATS if c.endswith(BEH)]
print(f'harmonised predictors: {len(FEATS)} (dropped {len(DROP)} Fahrenheit) '
      f'| physiology {len(PHYS)} | behaviour {len(BEHF)}')

g = dev.subject_id.values
tr_i, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
                  .split(dev, groups=g))
sub = dev.iloc[rest]
cal_r, te_r = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
                   .split(sub, groups=sub.subject_id.values))
cal_i, te_i = rest[cal_r], rest[te_r]
print(f'internal test: {len(te_i):,} instances, {int(dev.label.values[te_i].sum())} events')

res = {}
for name, cols in (('full', FEATS), ('physiology', PHYS), ('behaviour', BEHF)):
    ytr = dev.label.values[tr_i]
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                          random_state=SEED)
    m.fit(dev.iloc[tr_i][cols], ytr)
    p_cal = m.predict_proba(dev.iloc[cal_i][cols])[:, 1]
    platt = LogisticRegression(max_iter=1000).fit(logit(p_cal).reshape(-1, 1),
                                                  dev.label.values[cal_i])
    blk = {}
    for tag, frame, rows in (('internal', dev, te_i), ('external', ext, None)):
        X = frame.iloc[rows][cols] if rows is not None else frame[cols]
        y = frame.label.values[rows] if rows is not None else frame.label.values
        grp = (frame.subject_id.values[rows] if rows is not None
               else frame.subject_id.values)
        raw = m.predict_proba(X)[:, 1]
        cal = platt.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
        auc = roc_auc_score(y, raw)
        lo, hi = clustered_boot(y, raw, grp, roc_auc_score)
        sl, ic = slope_intercept(y, cal)
        slo, shi = clustered_boot(y, cal, grp, lambda yy, pp: slope_intercept(yy, pp)[0])
        blk[tag] = {'n': int(len(y)), 'events': int(y.sum()),
                    'prevalence_pct': round(100 * float(y.mean()), 3),
                    'auroc': round(float(auc), 4),
                    'auroc_ci': [round(lo, 4), round(hi, 4)],
                    'auroc_ci_width': round(hi - lo, 4),
                    'auprc': round(float(average_precision_score(y, raw)), 5),
                    'slope': round(sl, 4), 'slope_ci': [round(slo, 4), round(shi, 4)],
                    'intercept': round(ic, 4)}
    blk['drop'] = round(blk['internal']['auroc'] - blk['external']['auroc'], 4)
    res[name] = blk
    print(f"{name:<11} internal {blk['internal']['auroc']:.3f}  external "
          f"{blk['external']['auroc']:.3f} [{blk['external']['auroc_ci'][0]:.3f},"
          f"{blk['external']['auroc_ci'][1]:.3f}]  drop {blk['drop']:+.3f}  "
          f"slope {blk['external']['slope']:.2f} "
          f"[{blk['external']['slope_ci'][0]:.2f},{blk['external']['slope_ci'][1]:.2f}]  "
          f"int {blk['external']['intercept']:.2f}")
    del m; gc.collect()

json.dump(res, open(OUT + 'replicate.json', 'w'), indent=1)
print('\nwritten replicate.json')
