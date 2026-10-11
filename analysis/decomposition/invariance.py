#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two tests that decide how much weight the decomposition can carry.

(1) Is the behaviour share a property of the data or of gradient boosting?
    Repeat the decomposition with a linear model and a bagged tree ensemble.
(2) Does measurement behaviour carry information physiology does not, or is it
    a proxy for severity the physiological block already encodes? Stratify the
    test set by physiology-predicted risk and re-measure inside each stratum.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
XPAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
            tree_method='hist', n_jobs=4, verbosity=0)
H = '4h'
df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
beh = [c for c in feats if c.endswith(('_missing', '_time_since_obs_h'))]
phy = [c for c in feats if c not in beh]
rec = [c for c in feats if c.endswith('_time_since_obs_h')]
y = df['label'].values
g = df['subject_id'].values
med = df[feats].median()
OUT = {'model_class': {}, 'conditional': {}}


def run(model, cols, tr, te, scale=False):
    X = df[cols].fillna(med[cols])
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    if scale:
        s = StandardScaler().fit(Xtr)
        Xtr, Xte = s.transform(Xtr), s.transform(Xte)
    model.fit(Xtr, y[tr])
    p = model.predict_proba(Xte)[:, 1]
    return p, float(roc_auc_score(y[te], p))


res = {m: {b: [] for b in ('full', 'phys', 'beh')} for m in ('XGBoost', 'Logistic regression',
                                                             'Random forest')}
cond = {f'Q{i+1}': [] for i in range(5)}
cond_n = {f'Q{i+1}': [] for i in range(5)}

for s in range(5):
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s).split(df, y, g))
    pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
    p_phys_full = None
    for bname, cols in (('full', feats), ('phys', phy), ('beh', beh)):
        _, a = run(xgb.XGBClassifier(**XPAR, scale_pos_weight=pw, random_state=100 + s),
                   cols, tr, te)
        res['XGBoost'][bname].append(a)
        p, a2 = run(LogisticRegression(max_iter=400, class_weight='balanced', n_jobs=4),
                    cols, tr, te, scale=True)
        res['Logistic regression'][bname].append(a2)
        _, a3 = run(RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=20,
                                           class_weight='balanced_subsample', n_jobs=4,
                                           random_state=100 + s), cols, tr, te)
        res['Random forest'][bname].append(a3)
        if bname == 'phys':
            p_phys_full = None
        gc.collect()
        print(f'  split {s+1} block {bname} done', flush=True)

    # ---- conditional test: behaviour inside strata of physiology-predicted risk ----
    m = xgb.XGBClassifier(**XPAR, scale_pos_weight=pw, random_state=100 + s)
    m.fit(df.iloc[tr][phy], y[tr])
    risk = m.predict_proba(df.iloc[te][phy])[:, 1]
    del m; gc.collect()
    mb = xgb.XGBClassifier(**XPAR, scale_pos_weight=pw, random_state=100 + s)
    mb.fit(df.iloc[tr][beh], y[tr])
    pb = mb.predict_proba(df.iloc[te][beh])[:, 1]
    del mb; gc.collect()
    yt = y[te]
    cuts = np.quantile(risk, [.2, .4, .6, .8])
    strata = np.digitize(risk, cuts)
    for i in range(5):
        mk = strata == i
        if mk.sum() and yt[mk].sum() >= 10 and len(np.unique(yt[mk])) > 1:
            cond[f'Q{i+1}'].append(float(roc_auc_score(yt[mk], pb[mk])))
            cond_n[f'Q{i+1}'].append(int(yt[mk].sum()))
    print(f'  split {s+1} conditional done', flush=True)

for m in res:
    OUT['model_class'][m] = {b: dict(mean=float(np.mean(v)), sd=float(np.std(v)))
                             for b, v in res[m].items()}
    f, bh = np.array(res[m]['full']), np.array(res[m]['beh'])
    OUT['model_class'][m]['behaviour_share'] = float(np.mean((bh - .5) / (f - .5)))
for k in cond:
    if cond[k]:
        OUT['conditional'][k] = dict(auroc_mean=float(np.mean(cond[k])),
                                     auroc_sd=float(np.std(cond[k])),
                                     median_events=int(np.median(cond_n[k])),
                                     splits=len(cond[k]))
json.dump(OUT, open('/home/claude/rerun/invariance.json', 'w'), indent=2)
print(json.dumps(OUT, indent=2))
