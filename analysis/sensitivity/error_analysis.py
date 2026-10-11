#!/usr/bin/env python3
"""v6 Section 6 error-analysis plan: false-positive and false-negative profiling."""
import pandas as pd, numpy as np, json, warnings, gc
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

BASE = '/mnt/user-data/uploads/outputs/cleaned_data/'
df = pd.read_csv(BASE + 'engineered_features_4h_FULLSCALE.csv', low_memory=False)
feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
y = df['label'].values; g = df['subject_id'].values

tr_all, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(df, y, g))
sub = df.iloc[tr_all]
tr_i, cal_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=43)
                   .split(sub, y[tr_all], g[tr_all]))
tr, cal = tr_all[tr_i], tr_all[cal_i]
pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()

m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=pw, eval_metric='auc', tree_method='hist',
                      n_jobs=4, verbosity=0, random_state=42).fit(df.iloc[tr][feats], y[tr])


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


pc = m.predict_proba(df.iloc[cal][feats])[:, 1]
lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
lr.fit(logit(pc).reshape(-1, 1), y[cal])
p = lr.predict_proba(logit(m.predict_proba(df.iloc[te][feats])[:, 1]).reshape(-1, 1))[:, 1]

yt = y[te]; T = df.iloc[te].reset_index(drop=True)
thr = np.quantile(p[yt == 1], 0.30)        # 70% sensitivity operating point
flag = p >= thr
TP = flag & (yt == 1); FP = flag & (yt == 0)
FN = (~flag) & (yt == 1); TN = (~flag) & (yt == 0)
print(f'Operating point: 70% sensitivity, threshold {thr:.4f}')
print(f'  TP {TP.sum():,}   FP {FP.sum():,}   FN {FN.sum():,}   TN {TN.sum():,}')
print(f'  PPV {TP.sum()/max(flag.sum(),1):.4f}   flag rate {flag.mean():.4f}\n')

PROFILE = ['heart_rate_mean', 'heart_rate_max', 'resp_rate_mean', 'spo2_min',
           'mbp_nbp_mean', 'sbp_nbp_min', 'lactate_max', 'creatinine_mean',
           'wbc_mean', 'lactate_missing', 'sbp_abp_missing',
           'heart_rate_time_since_obs_h', 'lactate_time_since_obs_h']

print(f'{"Feature":<32}{"TP":>10}{"FP":>10}{"FN":>10}{"TN":>10}')
print('-' * 72)
rows = {}
for f in PROFILE:
    if f not in T.columns: continue
    v = [float(T.loc[mask, f].median()) if mask.sum() else float('nan')
         for mask in (TP, FP, FN, TN)]
    rows[f] = dict(TP=v[0], FP=v[1], FN=v[2], TN=v[3])
    print(f'{f:<32}{v[0]:>10.2f}{v[1]:>10.2f}{v[2]:>10.2f}{v[3]:>10.2f}')

print('\n--- False positives vs true positives: where do they differ least? ---')
diffs = []
for f, v in rows.items():
    if np.isnan(v['TP']) or np.isnan(v['FP']): continue
    sd = T[f].std()
    if sd and sd > 0:
        diffs.append((f, abs(v['TP'] - v['FP']) / sd))
diffs.sort(key=lambda x: x[1])
for f, d in diffs[:6]:
    print(f'  {f:<32} |TP-FP| = {d:.3f} SD')
print('  -> false positives resemble true positives most closely on these,')
print('     i.e. they are the patients the model cannot separate.')

print('\n--- False negatives: what did the model miss? ---')
for f, v in list(rows.items())[:8]:
    if np.isnan(v['FN']) or np.isnan(v['TP']): continue
    sd = T[f].std()
    if sd and sd > 0:
        print(f'  {f:<32} FN median {v["FN"]:>8.2f} vs TP {v["TP"]:>8.2f}  '
              f'({(v["FN"]-v["TP"])/sd:+.2f} SD)')

json.dump({'threshold': float(thr), 'counts': dict(TP=int(TP.sum()), FP=int(FP.sum()),
           FN=int(FN.sum()), TN=int(TN.sum())), 'profile': rows},
          open('/home/claude/sens/error_analysis.json', 'w'), indent=2)
print('\nwrote error_analysis.json')
