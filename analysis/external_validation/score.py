#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute the Appendix B protocol on the eICU demonstration subset.

Appendix B.2 requires frozen artefacts. The trained binaries from the original
run were not available, so the model is re-fitted to the specification recorded
in Chapter 3 and Section 4.8.4 on the MIMIC-IV development split, then frozen
before eICU is touched. The internal three-way-split figure is reported so the
re-fit can be checked against Table 4.11's 0.766.

eICU is scored exactly once per specification. No retraining on eICU, no
refitting of the recalibration, no threshold reselection.
"""
import numpy as np, pandas as pd, json, gc
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
OUT = '/home/claude/extval/'
SEED = 42
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
BEH = ('_missing', '_time_since_obs_h')


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def hanley_ci(auc, npos, nneg):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2)
                  + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg))
    return auc - 1.96 * se, auc + 1.96 * se, 2 * 1.96 * se


def cal_slope_intercept(p, y):
    """Cox calibration: logistic regression of y on logit(p)."""
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


dev = pd.read_csv(B + 'engineered_features_4h_FULLSCALE.csv', low_memory=False)
ext = pd.read_csv(OUT + 'eicu_features_4h.csv', low_memory=False)
FEATS = [c for c in dev.columns if c not in ('stay_id', 'subject_id', 'label')]
PHYS = [c for c in FEATS if not c.endswith(BEH)]
BEHF = [c for c in FEATS if c.endswith(BEH)]
assert len(FEATS) == 81 and len(PHYS) == 53 and len(BEHF) == 28
assert list(ext[FEATS].columns) == FEATS, 'column order must match'

# ----------------- three-way patient-level split of the development cohort ----
g = dev.subject_id.values
tr_i, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
                  .split(dev, groups=g))
sub = dev.iloc[rest]
cal_r, te_r = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
                   .split(sub, groups=sub.subject_id.values))
cal_i, te_i = rest[cal_r], rest[te_r]
print(f'development split  train {len(tr_i):,}  calibration {len(cal_i):,}  '
      f'internal test {len(te_i):,}')

res = {'cohort': json.load(open(OUT + 'eicu_cohort.json'))}
res['development'] = {'instances': int(len(dev)), 'events': int(dev.label.sum()),
                      'prevalence_pct': round(100 * float(dev.label.mean()), 4)}

for name, cols in (('full', FEATS), ('physiology', PHYS), ('behaviour', BEHF)):
    ytr = dev.label.values[tr_i]
    pw = (len(ytr) - ytr.sum()) / ytr.sum()
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=SEED)
    m.fit(dev.iloc[tr_i][cols], ytr)

    p_cal = m.predict_proba(dev.iloc[cal_i][cols])[:, 1]
    platt = LogisticRegression(max_iter=1000).fit(logit(p_cal).reshape(-1, 1),
                                                  dev.label.values[cal_i])

    def scored(X):
        raw = m.predict_proba(X)[:, 1]
        return raw, platt.predict_proba(logit(raw).reshape(-1, 1))[:, 1]

    # ---- internal test (the re-fit check against Table 4.11) ----
    yte = dev.label.values[te_i]
    raw_i, cal_in = scored(dev.iloc[te_i][cols])
    # ---- external, scored once ----
    yex = ext.label.values
    raw_e, cal_ex = scored(ext[cols])

    blk = {}
    for tag, y, raw, calp in (('internal', yte, raw_i, cal_in),
                              ('external', yex, raw_e, cal_ex)):
        auc = roc_auc_score(y, raw)
        lo, hi, w = hanley_ci(auc, int(y.sum()), int((1 - y).sum()))
        sl, ic = cal_slope_intercept(calp, y)
        slr, icr = cal_slope_intercept(raw, y)
        blk[tag] = {
            'n': int(len(y)), 'events': int(y.sum()),
            'prevalence_pct': round(100 * float(y.mean()), 4),
            'auroc': round(float(auc), 4),
            'auroc_ci': [round(float(lo), 4), round(float(hi), 4)],
            'auroc_ci_width': round(float(w), 4),
            'auprc': round(float(average_precision_score(y, raw)), 5),
            'auprc_lift_over_prevalence': round(
                float(average_precision_score(y, raw)) / float(y.mean()), 2),
            'slope_raw': round(slr, 4), 'intercept_raw': round(icr, 4),
            'slope_recalibrated': round(sl, 4), 'intercept_recalibrated': round(ic, 4),
        }
    blk['delta_auroc_external_minus_internal'] = round(
        blk['external']['auroc'] - blk['internal']['auroc'], 4)
    res[name] = blk
    print(f"{name:<11} internal {blk['internal']['auroc']:.3f}  "
          f"external {blk['external']['auroc']:.3f}  "
          f"delta {blk['delta_auroc_external_minus_internal']:+.3f}  "
          f"ext CI width {blk['external']['auroc_ci_width']:.3f}")
    del m; gc.collect()

# ------------------------------- block shares of above-chance discrimination ----
for tag in ('internal', 'external'):
    f = res['full'][tag]['auroc'] - 0.5
    res.setdefault('shares_pct', {})[tag] = {
        'physiology': round(100 * (res['physiology'][tag]['auroc'] - 0.5) / f, 1),
        'behaviour': round(100 * (res['behaviour'][tag]['auroc'] - 0.5) / f, 1),
    }
print('\nshares of above-chance discrimination:', res['shares_pct'])

json.dump(res, open(OUT + 'extval.json', 'w'), indent=1)
print('\nwritten extval.json')
