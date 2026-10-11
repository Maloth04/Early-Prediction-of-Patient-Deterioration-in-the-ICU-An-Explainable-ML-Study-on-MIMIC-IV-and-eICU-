#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attribution-rank agreement between the MIMIC-IV and eICU rankings.

Appendix B.1 requires it; Section 4.12.1 records that the executions did not
produce it. It is the one quantity on Table B.1's list that the 27 external
events do not bar, because mean absolute SHAP is computed over all 17,776
external instances and does not use the labels at all.

The question it answers is H3's, asked across institutions rather than across
horizons: does the model explain itself the same way in a second hospital system?
Internally, rank agreement within a horizon is 0.996 (Table 4.19); across
horizons it falls to 0.528 (Section 4.6), which is why H3 is only partially
supported. This places the institutional comparison on that scale.

TreeSHAP is exact for XGBoost, so no sampling error enters the attributions; the
interval below is over patient-clustered resamples of the instances.
"""
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
OUT = '/home/claude/extval/'
SEED, NBOOT = 42, 500
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
BEH = ('_missing', '_time_since_obs_h')
rng = np.random.default_rng(SEED)
res = {}

for H in ('4h', '8h'):
    dev = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    ext = pd.read_csv(OUT + f'eicu_features_{H}.csv', low_memory=False)
    ALL = [c for c in dev.columns if c not in ('stay_id', 'subject_id', 'label')]
    F = [c for c in ALL if not c.startswith('temp_f_')]          # the 75 harmonised
    if 'sid' in ext.columns:                      # the 8-hour matrix keys on sid
        ext = ext.rename(columns={'sid': 'stay_id'})
    if 'subject_id' not in ext.columns:           # one unit stay = one patient in eICU
        ext['subject_id'] = ext['stay_id']
    missing = [c for c in F if c not in ext.columns]
    assert not missing, f'external matrix lacks {missing[:5]}'

    g = dev.subject_id.values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED).split(dev, groups=g))
    ytr = dev.label.values[tr]
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                          random_state=SEED)
    m.fit(dev.iloc[tr][F], ytr)
    auc_i = roc_auc_score(dev.label.values[te], m.predict_proba(dev.iloc[te][F])[:, 1])
    auc_e = roc_auc_score(ext.label.values, m.predict_proba(ext[F])[:, 1])

    expl = shap.TreeExplainer(m)
    Xi = dev.iloc[te][F]
    Xe = ext[F]
    si = np.abs(expl.shap_values(Xi)).mean(axis=0)
    se = np.abs(expl.shap_values(Xe)).mean(axis=0)

    rho = spearmanr(si, se).statistic
    oi = [F[i] for i in np.argsort(-si)]
    oe = [F[i] for i in np.argsort(-se)]
    top15 = len(set(oi[:15]) & set(oe[:15]))
    top5 = len(set(oi[:5]) & set(oe[:5]))

    # patient-clustered bootstrap of the external attributions
    gq = ext.subject_id.values
    uq = np.unique(gq)
    idx = {q: np.flatnonzero(gq == q) for q in uq}
    sv_e = np.abs(expl.shap_values(Xe))
    boot = []
    for _ in range(NBOOT):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idx[q] for q in pick])
        boot.append(spearmanr(si, sv_e[ii].mean(axis=0)).statistic)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # where does the disagreement sit? behaviour vs physiology
    isbeh = np.array([c.endswith(BEH) for c in F])
    rho_phys = spearmanr(si[~isbeh], se[~isbeh]).statistic
    rho_beh = spearmanr(si[isbeh], se[isbeh]).statistic
    shr_i = si[isbeh].sum() / si.sum()
    shr_e = se[isbeh].sum() / se.sum()

    res[H] = {'internal_auroc': round(float(auc_i), 4), 'external_auroc': round(float(auc_e), 4),
              'spearman_rho': round(float(rho), 4), 'rho_ci': [round(lo, 4), round(hi, 4)],
              'top5_overlap': top5, 'top15_overlap': top15,
              'rho_physiology_block': round(float(rho_phys), 4),
              'rho_behaviour_block': round(float(rho_beh), 4),
              'behaviour_share_of_attribution_internal': round(float(shr_i), 4),
              'behaviour_share_of_attribution_external': round(float(shr_e), 4),
              'top10_internal': oi[:10], 'top10_external': oe[:10]}
    r = res[H]
    print(f'=== {H} ===')
    print(f'  AUROC internal {auc_i:.3f}  external {auc_e:.3f}')
    print(f'  rank agreement rho = {rho:.3f} [{lo:.3f}, {hi:.3f}]   '
          f'top-5 overlap {top5}/5, top-15 {top15}/15')
    print(f'  within physiology block rho={rho_phys:.3f} | within behaviour block rho={rho_beh:.3f}')
    print(f'  behaviour share of total |SHAP|: internal {shr_i:.1%} -> external {shr_e:.1%}')
    print(f'  internal top 5: {oi[:5]}')
    print(f'  external top 5: {oe[:5]}')
    print()

json.dump(res, open(OUT + 'shap_transport.json', 'w'), indent=1)
print('written shap_transport.json')
