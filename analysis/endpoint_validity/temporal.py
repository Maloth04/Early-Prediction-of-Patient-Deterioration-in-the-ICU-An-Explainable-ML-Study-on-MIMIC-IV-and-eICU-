#!/usr/bin/env python3
"""H2a executed: train on earlier admission years, evaluate on later ones.

The comparison that matters is temporal-split performance against a
contemporaneous random split of the SAME size, so any drop is attributable to
the passage of time rather than to having less training data.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
EARLY = ('2008 - 2010', '2011 - 2013')
LATE = ('2014 - 2016', '2017 - 2019')


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def calib(y, p):
    lp = logit(p).reshape(-1, 1)
    slope = float(LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
                  .fit(lp, y).coef_[0][0])
    off = logit(p); lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(off + mid)))).mean() < y.mean(): lo = mid
        else: hi = mid
    return slope, float((lo + hi) / 2)


pat = pd.read_csv(B + 'patients_CLEANED.csv', usecols=['subject_id', 'anchor_year_group'],
                  low_memory=False)
OUT = {}
for H in ('4h', '8h'):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    df = df.merge(pat, on='subject_id', how='left')
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label', 'anchor_year_group')]
    y = df['label'].values
    early = df.anchor_year_group.isin(EARLY).values
    late = df.anchor_year_group.isin(LATE).values
    print(f'[{H}] early {early.sum():,} inst / {int(y[early].sum()):,} ev   '
          f'late {late.sum():,} inst / {int(y[late].sum()):,} ev')

    # ---- temporal split ----
    pw = (early.sum() - y[early].sum()) / y[early].sum()
    m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=21)
    m.fit(df.loc[early, feats], y[early])
    p_late = m.predict_proba(df.loc[late, feats])[:, 1]
    t_auc = roc_auc_score(y[late], p_late)
    t_ap = average_precision_score(y[late], p_late)
    t_sl, t_in = calib(y[late], p_late)
    del m; gc.collect()

    # ---- contemporaneous reference, matched training size, 5 seeds ----
    ref = {'auc': [], 'ap': [], 'sl': []}
    n_tr = int(early.sum())
    for rep in range(5):
        g = df['subject_id'].values
        frac = 1 - n_tr / len(df)
        tr, te = next(GroupShuffleSplit(1, test_size=frac, random_state=40 + rep)
                      .split(df, y, g))
        pw2 = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
        m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw2, random_state=40 + rep)
        m.fit(df.iloc[tr][feats], y[tr])
        p = m.predict_proba(df.iloc[te][feats])[:, 1]
        ref['auc'].append(roc_auc_score(y[te], p))
        ref['ap'].append(average_precision_score(y[te], p))
        ref['sl'].append(calib(y[te], p)[0])
        del m; gc.collect()
        print(f'   reference rep {rep+1}/5', flush=True)

    r_auc, r_sd = float(np.mean(ref['auc'])), float(np.std(ref['auc']))
    drop = r_auc - t_auc
    OUT[H] = dict(temporal_auroc=float(t_auc), temporal_auprc=float(t_ap),
                  temporal_slope=t_sl, temporal_intercept=t_in,
                  ref_auroc=r_auc, ref_sd=r_sd,
                  ref_auprc=float(np.mean(ref['ap'])), ref_slope=float(np.mean(ref['sl'])),
                  drop=float(drop),
                  n_train=int(early.sum()), n_test=int(late.sum()),
                  ev_train=int(y[early].sum()), ev_test=int(y[late].sum()))
    print(f'  TEMPORAL  AUROC {t_auc:.3f}  AUPRC {t_ap:.4f}  slope {t_sl:.3f}')
    print(f'  REFERENCE AUROC {r_auc:.3f} +/- {r_sd:.3f}  AUPRC {np.mean(ref["ap"]):.4f}  '
          f'slope {np.mean(ref["sl"]):.3f}')
    print(f'  DROP {drop:+.3f}   criterion is a drop of <= 0.05  -> '
          f'{"MET" if drop <= 0.05 else "NOT MET"}\n')
    del df; gc.collect()

json.dump(OUT, open('/home/claude/temporal/temporal.json', 'w'), indent=2)
print('wrote temporal.json')
