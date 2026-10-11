#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The registered H3 statistic for the between-institution comparison.

Section 4.12.4 argued the eight-hour explanation 'fails' by citing Table 4.19's
0.859 as evidence that Spearman's rho is insensitive to head reordering. That
reasoning is wrong twice: a fall to 0.859 shows the measure is SENSITIVE, and
Table 4.19's statistic is over 15 features where Section 4.12.4's is over 75, so
the two are not comparable. The registered criterion is rho over the union of
the two top-fifteen lists against 0.7 (Section 4.6). It was never computed for
this comparison. It is computed here, and it decides the claim.

Also computed, because mean absolute SHAP can be large and nearly constant for a
feature that separates nobody: the same quantities on centred attributions,
mean|SHAP - mean(SHAP)|, which charges a feature only for the variation it
contributes. eICU's continuous-channel recency features have an inter-quartile
range of 3 minutes (Table 4.42), so they are exactly the case at risk.
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


def union_rho(si, se, F, k=15):
    """Spearman's rho over the union of the two top-k lists: the registered statistic."""
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

    g = dev.subject_id.values
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
                  .split(dev, groups=g))
    ytr = dev.label.values[tr]
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(),
                          random_state=SEED)
    m.fit(dev.iloc[tr][F], ytr)
    expl = shap.TreeExplainer(m)
    SVi = expl.shap_values(dev.iloc[te][F])
    SVe = expl.shap_values(ext[F])

    out = {}
    for tag, fn in (('mean_abs', lambda S: np.abs(S).mean(axis=0)),
                    ('centred', lambda S: np.abs(S - S.mean(axis=0)).mean(axis=0))):
        si, se = fn(SVi), fn(SVe)
        rho_all = float(spearmanr(si, se).statistic)
        ru, nu = union_rho(si, se, F)
        oi = [F[i] for i in np.argsort(-si)]; oe = [F[i] for i in np.argsort(-se)]
        isbeh = np.array([c.endswith(BEH) for c in F])
        out[tag] = {'rho_all75': round(rho_all, 4),
                    'rho_union_top15': round(ru, 4), 'union_size': nu,
                    'top5_overlap': len(set(oi[:5]) & set(oe[:5])),
                    'top15_overlap': len(set(oi[:15]) & set(oe[:15])),
                    'beh_share_internal': round(float(si[isbeh].sum() / si.sum()), 4),
                    'beh_share_external': round(float(se[isbeh].sum() / se.sum()), 4),
                    'top5_internal': oi[:5], 'top5_external': oe[:5]}
        print(f'=== {H} / {tag} ===')
        print(f'  rho over all 75          {rho_all:.4f}')
        print(f'  rho over union of top-15 {ru:.4f}   (union = {nu} features)  '
              f'vs registered 0.7 -> {"PASS" if ru >= 0.7 else "FAIL"}')
        print(f'  top-5 overlap {out[tag]["top5_overlap"]}/5, top-15 {out[tag]["top15_overlap"]}/15')
        print(f'  behaviour share  internal {out[tag]["beh_share_internal"]:.1%} '
              f'-> external {out[tag]["beh_share_external"]:.1%}')
        print(f'  internal top5 {oi[:5]}')
        print(f'  external top5 {oe[:5]}')

    # clustered bootstrap of the registered statistic, external cohort resampled
    si = np.abs(SVi).mean(axis=0)
    gq = ext.subject_id.values; uq = np.unique(gq)
    idx = {q: np.flatnonzero(gq == q) for q in uq}
    A = np.abs(SVe)
    boot = []
    for _ in range(NBOOT):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([idx[q] for q in pick])
        boot.append(union_rho(si, A[ii].mean(axis=0), F)[0])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    frac = float(np.mean(np.array(boot) >= 0.7))
    out['union_ci_mean_abs'] = [round(lo, 4), round(hi, 4)]
    out['union_frac_above_0.7'] = round(frac, 4)
    print(f'  registered statistic, clustered bootstrap: [{lo:.3f}, {hi:.3f}], '
          f'{frac:.1%} of resamples at or above 0.7\n')
    res[H] = out

json.dump(res, open(OUT + 'h3_union.json', 'w'), indent=1)
print('written h3_union.json')
