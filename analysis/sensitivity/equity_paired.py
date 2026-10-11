#!/usr/bin/env python3
"""Paired subgroup test: is the deficit consistent in DIRECTION across splits?

Comparing a subgroup mean against an overall mean across independent runs conflates
the subgroup effect with split-to-split noise. The right test pairs them: on each
split, compute (subgroup AUROC - White AUROC) on the SAME fitted model, then ask
whether the sign is consistent. This is what distinguishes a real deficit from the
tail draw that produced the withdrawn Asian-subgroup finding.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
import xgboost as xgb

BASE = '/mnt/user-data/uploads/outputs/cleaned_data/'
df = pd.read_csv(BASE + 'engineered_features_4h_FULLSCALE.csv', low_memory=False)
feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
y = df['label'].values; g = df['subject_id'].values

adm = pd.read_csv('/mnt/user-data/uploads/admissions.csv',
                  usecols=['subject_id', 'race'], low_memory=False)
rm = adm.drop_duplicates('subject_id').set_index('subject_id')['race']


def bucket(r):
    if not isinstance(r, str): return 'Unknown'
    r = r.upper()
    if 'WHITE' in r: return 'White'
    if 'BLACK' in r: return 'Black'
    if 'HISPANIC' in r or 'LATINO' in r: return 'Hispanic'
    if 'ASIAN' in r: return 'Asian'
    if 'UNKNOWN' in r or 'UNABLE' in r or 'DECLINED' in r: return 'Unknown'
    return 'Other'


df['race_g'] = df['subject_id'].map(rm).map(bucket)
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
LEVELS = ['Black', 'Hispanic', 'Asian', 'Other']
N_REP = 10
deltas = {l: [] for l in LEVELS}
counts = {l: [] for l in LEVELS}
white_auc = []

for rep in range(N_REP):
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=500 + rep).split(df, y, g))
    pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=500 + rep)
    m.fit(df.iloc[tr][feats], y[tr])
    p = m.predict_proba(df.iloc[te][feats])[:, 1]
    T = df.iloc[te]; yt = y[te]
    wm = (T['race_g'] == 'White').values
    aw = roc_auc_score(yt[wm], p[wm])
    white_auc.append(aw)
    for l in LEVELS:
        mask = (T['race_g'] == l).values
        ne = int(yt[mask].sum())
        if mask.sum() < 500 or ne < 15:
            continue
        deltas[l].append(roc_auc_score(yt[mask], p[mask]) - aw)
        counts[l].append(ne)
    del m; gc.collect()
    print(f'  rep {rep+1}/{N_REP}', flush=True)

print(f'\nWhite reference AUROC: mean {np.mean(white_auc):.3f} (SD {np.std(white_auc):.3f})\n')
print(f'{"Subgroup":<11}{"reps":>6}{"med events":>12}{"mean d":>9}{"SD":>8}'
      f'{"neg/total":>11}{"sign p":>9}')
OUT = {}
for l in LEVELS:
    d = np.array(deltas[l])
    if len(d) == 0:
        print(f'{l:<11}{"0":>6}   never met the minimum event threshold'); continue
    neg = int((d < 0).sum())
    pv = binomtest(neg, len(d), 0.5, alternative='greater').pvalue
    OUT[l] = dict(n_reps=len(d), median_events=float(np.median(counts[l])),
                  mean_delta=float(d.mean()), sd=float(d.std()),
                  n_negative=neg, sign_p=float(pv),
                  ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)))
    print(f'{l:<11}{len(d):>6}{np.median(counts[l]):>12.0f}{d.mean():>+9.3f}'
          f'{d.std():>8.3f}{f"{neg}/{len(d)}":>11}{pv:>9.4f}')

print('\nInterpretation guide:')
print('  sign p is a one-sided binomial test that the deficit direction is consistent.')
print('  A large mean deficit with an inconsistent sign is noise; a small deficit with')
print('  a consistent sign across independent splits is a real, if modest, effect.')
print('  Four subgroups are tested, so a Bonferroni-corrected threshold is p < 0.0125.')
json.dump({'white_mean': float(np.mean(white_auc)), 'subgroups': OUT},
          open('/home/claude/final/equity_paired.json', 'w'), indent=2)
print('\nwrote equity_paired.json')
