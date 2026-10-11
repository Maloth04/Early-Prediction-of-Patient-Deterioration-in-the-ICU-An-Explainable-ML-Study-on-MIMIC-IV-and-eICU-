# -*- coding: utf-8 -*-
"""Table 4.27's combined row disagreed with Table 4.26 by 0.004 because the
sub-block run passed the same 28 columns in a different order, and colsample_bytree
samples columns by position. Recompute preserving the original column order."""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
OUT = {}
for H in ('4h', '8h'):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    beh = [c for c in feats if c.endswith(('_missing', '_time_since_obs_h'))]   # document order
    rec = [c for c in beh if c.endswith('_time_since_obs_h')]
    miss = [c for c in beh if c.endswith('_missing')]
    y = df['label'].values; g = df['subject_id'].values
    res = {k: [] for k in ('recency', 'presence', 'both')}
    for s in range(5):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s).split(df, y, g))
        for name, cols in (('recency', rec), ('presence', miss), ('both', beh)):
            pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
            m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=100 + s)
            m.fit(df.iloc[tr][cols], y[tr])
            res[name].append(float(roc_auc_score(y[te], m.predict_proba(df.iloc[te][cols])[:, 1])))
            del m; gc.collect()
        print(f'  [{H}] split {s+1}/5', flush=True)
    OUT[H] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v))) for k, v in res.items()}
    print(H, {k: round(v['mean'], 4) for k, v in OUT[H].items()}, flush=True)
    del df; gc.collect()
json.dump(OUT, open('/home/claude/rerun/sub.json', 'w'), indent=2)
print('wrote sub.json')
