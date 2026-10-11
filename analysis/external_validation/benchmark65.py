#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The amended H2b benchmark, computed so that Amendment 6 of Appendix B.6 names
a figure that exists.

Appendix B.4 moves the benchmark from the full model to 'the physiological set
augmented by the laboratory and cuff-pressure behaviour features' and gives the
count as 65. That count is in the 81-predictor space. The external comparison
runs in the 75-predictor harmonised space, where the same set is 61 predictors,
because temperature's six predictors are unavailable in the extraction. Both are
computed here on the split behind Table 4.41, so a future executor has the
figure in whichever space the extraction supports.
"""
import numpy as np, pandas as pd, json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
SEED = 42
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
BEH = ('_missing', '_time_since_obs_h')
LAB = ('creatinine', 'lactate', 'wbc')
CUFF = ('sbp_nbp', 'dbp_nbp', 'mbp_nbp')
out = {}

for H in ('4h', '8h'):
    dev = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    F81 = [c for c in dev.columns if c not in ('stay_id', 'subject_id', 'label')]
    F75 = [c for c in F81 if not c.startswith('temp_f_')]
    assert len(F81) == 81 and len(F75) == 75

    def sets(F):
        phys = [c for c in F if not c.endswith(BEH)]
        beh = [c for c in F if c.endswith(BEH)]
        keep = [c for c in beh if c.startswith(LAB) or c.startswith(CUFF)]
        return phys, beh, phys + keep

    g = dev.subject_id.values
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
                    .split(dev, groups=g))
    sub = dev.iloc[rest]
    _, te_r = next(GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
                   .split(sub, groups=sub.subject_id.values))
    te = rest[te_r]
    ytr, yte = dev.label.values[tr], dev.label.values[te]
    print(f'=== {H}: train {len(tr):,}  internal test {len(te):,} instances, '
          f'{int(yte.sum())} events')
    out[H] = {'test_instances': int(len(te)), 'test_events': int(yte.sum())}

    for space, F in (('81', F81), ('75', F75)):
        phys, beh, bench = sets(F)
        for name, cols in (('full', F), ('benchmark', bench), ('physiology', phys)):
            m = xgb.XGBClassifier(**PAR, scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                                  random_state=SEED)
            m.fit(dev.iloc[tr][cols], ytr)
            p = m.predict_proba(dev.iloc[te][cols])[:, 1]
            a = roc_auc_score(yte, p)
            out[H][f'{space}_{name}'] = {'n_predictors': len(cols), 'auroc': round(float(a), 4),
                                         'auprc': round(float(average_precision_score(yte, p)), 4)}
            print(f'  {space}-space {name:>10}  {len(cols):>2} predictors  AUROC {a:.4f}')

json.dump(out, open('/home/claude/extval/benchmark65.json', 'w'), indent=1)
print('\nwritten benchmark65.json')
