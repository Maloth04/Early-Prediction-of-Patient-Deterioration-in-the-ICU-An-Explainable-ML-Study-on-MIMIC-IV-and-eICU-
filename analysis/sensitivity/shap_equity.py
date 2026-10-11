#!/usr/bin/env python3
"""H3 (SHAP rank stability) and the equity analysis, both on the same fitted model.

On H3: a reported Spearman rho of 0.996 is suspiciously high. This script also tests
whether the metric is SENSITIVE ENOUGH to detect instability, by measuring what rho a
deliberately perturbed ranking produces. A metric that cannot fall is not evidence.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb, shap

BASE = '/mnt/user-data/uploads/outputs/cleaned_data/'
R = {}

df = pd.read_csv(BASE + f'engineered_features_4h_FULLSCALE.csv', low_memory=False)
feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
y = df['label'].values; g = df['subject_id'].values
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)

# ================= H3: SHAP RANK STABILITY =================
print('=== H3: SHAP rank stability, 200 patient-level bootstrap resamples ===')
tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=5).split(df, y, g))
pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
base = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=5).fit(df.iloc[tr][feats], y[tr])

# reference ranking on a fixed evaluation sample
rng = np.random.RandomState(5)
eval_idx = rng.choice(te, size=min(20000, len(te)), replace=False)
expl = shap.TreeExplainer(base)
sv = np.abs(expl.shap_values(df.loc[eval_idx, feats])).mean(axis=0)
ref = pd.Series(sv, index=feats).sort_values(ascending=False)
TOP = list(ref.head(15).index)
print('  reference top-10:', ', '.join(TOP[:10]))

pat = df.loc[te, 'subject_id'].unique()
rhos, top_sets = [], []
N_BOOT = 200
for b in range(N_BOOT):
    rb = np.random.RandomState(1000 + b)
    samp = rb.choice(pat, size=len(pat), replace=True)
    idx = df.index[df['subject_id'].isin(set(samp)) & df.index.isin(te)]
    if len(idx) > 20000:
        idx = rb.choice(idx, size=20000, replace=False)
    s = np.abs(expl.shap_values(df.loc[idx, feats])).mean(axis=0)
    ser = pd.Series(s, index=feats)
    union = list(dict.fromkeys(TOP + list(ser.sort_values(ascending=False).head(15).index)))
    rho = spearmanr(ref[union].rank(), ser[union].rank()).correlation
    rhos.append(rho)
    top_sets.append(set(ser.sort_values(ascending=False).head(15).index))
    if (b + 1) % 50 == 0:
        print(f'    {b+1}/{N_BOOT}', flush=True)

rhos = np.array(rhos)
R['h3'] = dict(mean=float(rhos.mean()), sd=float(rhos.std()),
               p5=float(np.percentile(rhos, 5)), minimum=float(rhos.min()),
               n_boot=N_BOOT)
print(f'\n  Spearman rho: mean {rhos.mean():.4f}  SD {rhos.std():.4f}  '
      f'5th pct {np.percentile(rhos,5):.4f}  min {rhos.min():.4f}')

# top-15 membership churn: a harder test than rank correlation
jac = [len(s & set(TOP)) / len(s | set(TOP)) for s in top_sets]
always = set(TOP)
for s in top_sets: always &= s
R['h3'].update(jaccard_mean=float(np.mean(jac)),
               n_always_in_top15=int(len(always)))
print(f'  top-15 Jaccard vs reference: mean {np.mean(jac):.3f}')
print(f'  features in the top 15 in ALL {N_BOOT} resamples: {len(always)} of 15')

# ---- IS THE METRIC SENSITIVE? deliberately perturb the ranking ----
print('\n  --- sensitivity of the metric itself ---')
sens = {}
for nswap in (1, 2, 3, 5):
    vals = []
    for t in range(200):
        rt = np.random.RandomState(t)
        perm = list(range(len(TOP)))
        for _ in range(nswap):
            i, j = rt.choice(len(perm), 2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
        vals.append(spearmanr(range(len(TOP)), perm).correlation)
    sens[f'{nswap} random swaps in the top-15'] = float(np.mean(vals))
    print(f'    {nswap} swap(s) -> rho {np.mean(vals):.4f}')
R['h3']['metric_sensitivity'] = sens
del expl; gc.collect()

# ================= EQUITY ANALYSIS =================
print('\n=== Equity analysis: subgroup discrimination, replicated ===')
adm = pd.read_csv('/mnt/user-data/uploads/admissions.csv',
                  usecols=['subject_id', 'race'], low_memory=False)
race_map = adm.drop_duplicates('subject_id').set_index('subject_id')['race']


def bucket(r):
    if not isinstance(r, str): return 'Unknown'
    r = r.upper()
    if 'WHITE' in r: return 'White'
    if 'BLACK' in r: return 'Black'
    if 'HISPANIC' in r or 'LATINO' in r: return 'Hispanic'
    if 'ASIAN' in r: return 'Asian'
    if 'UNKNOWN' in r or 'UNABLE' in r or 'DECLINED' in r: return 'Unknown'
    return 'Other'


df['race_g'] = df['subject_id'].map(race_map).map(bucket)
icu = pd.read_csv(BASE + 'icustays_CLEANED.csv', usecols=['stay_id', 'gender'], low_memory=False)
df['sex'] = df['stay_id'].map(icu.drop_duplicates('stay_id').set_index('stay_id')['gender'])
print(df['race_g'].value_counts().to_string())

GROUPS = [('race_g', ['White', 'Black', 'Hispanic', 'Asian', 'Other']),
          ('sex', ['M', 'F'])]
N_REP = 10
acc = {}
for rep in range(N_REP):
    trr, tee = next(GroupShuffleSplit(1, test_size=0.25, random_state=300 + rep).split(df, y, g))
    pwr = (len(y[trr]) - y[trr].sum()) / y[trr].sum()
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=pwr, random_state=300 + rep)
    m.fit(df.iloc[trr][feats], y[trr])
    p = m.predict_proba(df.iloc[tee][feats])[:, 1]
    T = df.iloc[tee]; yt = y[tee]
    acc.setdefault('Overall', []).append(roc_auc_score(yt, p))
    for col, levels in GROUPS:
        for lv in levels:
            mask = (T[col] == lv).values
            if mask.sum() < 500 or yt[mask].sum() < 15:
                continue
            acc.setdefault(f'{lv}', []).append(roc_auc_score(yt[mask], p[mask]))
    del m; gc.collect()
    print(f'  rep {rep+1}/{N_REP}', flush=True)

print(f'\n  {"Subgroup":<12}{"n reps":>8}{"mean":>9}{"SD":>8}{"min":>8}{"max":>8}{"vs overall":>12}')
ov = np.mean(acc['Overall'])
eq = {}
for kk, v in acc.items():
    v = np.array(v)
    eq[kk] = dict(mean=float(v.mean()), sd=float(v.std()), min=float(v.min()),
                  max=float(v.max()), n=len(v), delta=float(v.mean() - ov))
    print(f'  {kk:<12}{len(v):>8}{v.mean():>9.3f}{v.std():>8.3f}{v.min():>8.3f}'
          f'{v.max():>8.3f}{v.mean()-ov:>+12.3f}')
R['equity'] = eq

json.dump(R, open('/home/claude/final/shap_equity.json', 'w'), indent=2, default=str)
print('\nwrote shap_equity.json')
