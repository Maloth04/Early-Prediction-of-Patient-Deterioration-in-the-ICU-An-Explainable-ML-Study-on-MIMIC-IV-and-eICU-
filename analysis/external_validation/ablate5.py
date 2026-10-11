#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replicate the Section 4.12.3 feature-removal measurements across five splits.

The single-split figures were incoherent: dropping four features cost 0.017
while dropping those four plus four more cost 0.013. A subset cannot cost more
than its superset except by noise, and the thesis revised a registered benchmark
and withdrew a falsification line on that figure. This re-runs the removals on
the five patient-level splits used for Table 4.26 and reports means with spreads,
so the revision either stands on a replicated estimate or is withdrawn.
"""
import numpy as np, pandas as pd, json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
BEH = ('_missing', '_time_since_obs_h')
CONT = ['heart_rate', 'resp_rate', 'spo2', 'temp_c']      # Table 4.43's sub-block
NSPLIT = 5

dev = pd.read_csv(B + 'engineered_features_4h_FULLSCALE.csv', low_memory=False)
ALL = [c for c in dev.columns if c not in ('stay_id', 'subject_id', 'label')]
HARM = [c for c in ALL if not c.startswith('temp_f_')]
rec = [f'{v}_time_since_obs_h' for v in CONT]
both = rec + [f'{v}_missing' for v in CONT]
SETS = {
    'full_75': HARM,
    'minus_4_continuous_recency': [c for c in HARM if c not in rec],
    'minus_8_continuous_behaviour': [c for c in HARM if c not in both],
    'minus_26_behaviour': [c for c in HARM if not c.endswith(BEH)],
}
y_all = dev.label.values
g = dev.subject_id.values
res = {k: [] for k in SETS}
for s in range(NSPLIT):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=s).split(dev, groups=g))
    ytr, yte = y_all[tr], y_all[te]
    pw = (len(ytr) - ytr.sum()) / ytr.sum()
    for name, cols in SETS.items():
        m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=s)
        m.fit(dev.iloc[tr][cols], ytr)
        a = float(roc_auc_score(yte, m.predict_proba(dev.iloc[te][cols])[:, 1]))
        res[name].append(a)
    print(f'split {s}: ' + '  '.join(f'{k.split("_")[1]}={res[k][-1]:.4f}' for k in SETS), flush=True)

out = {}
base = np.array(res['full_75'])
print(f'\n{"configuration":<34}{"AUROC mean +/- SD":>22}{"drop vs full, paired":>26}')
for name in SETS:
    a = np.array(res[name])
    d = base - a
    out[name] = {'auroc_mean': round(float(a.mean()), 4), 'auroc_sd': round(float(a.std(ddof=1)), 4),
                 'drop_mean': round(float(d.mean()), 4), 'drop_sd': round(float(d.std(ddof=1)), 4),
                 'drop_min': round(float(d.min()), 4), 'drop_max': round(float(d.max()), 4),
                 'splits_positive': int((d > 0).sum()), 'per_split': [round(float(x), 4) for x in a]}
    o = out[name]
    print(f'{name:<34}{o["auroc_mean"]:.4f} +/- {o["auroc_sd"]:.4f}      '
          f'{o["drop_mean"]:+.4f} +/- {o["drop_sd"]:.4f}  ({o["splits_positive"]}/5 positive, '
          f'range {o["drop_min"]:+.4f} to {o["drop_max"]:+.4f})')

# is the 4-feature removal distinguishable from the 8-feature one?
d4 = base - np.array(res['minus_4_continuous_recency'])
d8 = base - np.array(res['minus_8_continuous_behaviour'])
diff = d4 - d8
print(f'\n4-feature drop minus 8-feature drop: {diff.mean():+.4f} +/- {diff.std(ddof=1):.4f} '
      f'({int((diff > 0).sum())}/5 splits positive)')
out['subset_vs_superset'] = {'mean': round(float(diff.mean()), 4),
                             'sd': round(float(diff.std(ddof=1)), 4),
                             'splits_positive': int((diff > 0).sum())}
json.dump(out, open('/home/claude/extval/ablate5.json', 'w'), indent=1)
print('\nwritten ablate5.json')
