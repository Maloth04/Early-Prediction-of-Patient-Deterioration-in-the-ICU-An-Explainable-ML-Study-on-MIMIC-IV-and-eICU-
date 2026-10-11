#!/usr/bin/env python3
"""Decision curve analysis against the incumbent decision-maker.

Standard DCA compares a model to treat-all and treat-none. Neither describes an ICU:
the real alternative is acting on the care team's own revealed concern. The
behaviour-only model operationalises that, so it is added as a third comparator.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

BASE = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def net_benefit(y, p, t):
    flag = p >= t
    n = len(y)
    tp = np.sum(flag & (y == 1)); fp = np.sum(flag & (y == 0))
    return tp / n - (fp / n) * (t / (1 - t))


def treat_all_nb(y, t):
    pi = y.mean()
    return pi - (1 - pi) * (t / (1 - t))


RES = {}
for H in ('4h', '8h'):
    df = pd.read_csv(BASE + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    allf = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    BEH = [c for c in allf if c.endswith('_missing') or c.endswith('_time_since_obs_h')]
    y = df['label'].values; g = df['subject_id'].values

    reps = []
    for rep in range(5):
        tr_all, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=700 + rep)
                          .split(df, y, g))
        sub = df.iloc[tr_all]
        tri, cali = next(GroupShuffleSplit(1, test_size=0.25, random_state=800 + rep)
                         .split(sub, y[tr_all], g[tr_all]))
        tr, cal = tr_all[tri], tr_all[cali]
        pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()

        out = {}
        for name, fs in (('full', allf), ('behaviour', BEH)):
            m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=700 + rep)
            m.fit(df.iloc[tr][fs], y[tr])
            pc = m.predict_proba(df.iloc[cal][fs])[:, 1]
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            lr.fit(logit(pc).reshape(-1, 1), y[cal])
            praw = m.predict_proba(df.iloc[te][fs])[:, 1]
            out[name] = lr.predict_proba(logit(praw).reshape(-1, 1))[:, 1]
            del m; gc.collect()
        reps.append((y[te], out))
        print(f'  {H} rep {rep+1}/5', flush=True)

    THRESH = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    rows = []
    for t in THRESH:
        nb_f = np.mean([net_benefit(yt, o['full'], t) for yt, o in reps])
        nb_b = np.mean([net_benefit(yt, o['behaviour'], t) for yt, o in reps])
        nb_a = np.mean([treat_all_nb(yt, t) for yt, o in reps])
        wins = sum(1 for yt, o in reps
                   if net_benefit(yt, o['full'], t) > net_benefit(yt, o['behaviour'], t))
        rows.append(dict(threshold=t, nb_full=nb_f, nb_behaviour=nb_b,
                         nb_treat_all=nb_a, nb_treat_none=0.0,
                         full_beats_behaviour=f'{wins}/5',
                         delta=nb_f - nb_b))
    RES[H] = rows
    print(f'\n=== {H}: net benefit (mean of 5 patient-level splits) ===')
    print(f'{"Thresh":>8}{"Full":>11}{"Behaviour":>12}{"Treat-all":>12}'
          f'{"Treat-none":>12}{"Full-Behav":>12}{"wins":>7}')
    for r in rows:
        print(f'{r["threshold"]:>8.3f}{r["nb_full"]:>11.5f}{r["nb_behaviour"]:>12.5f}'
              f'{r["nb_treat_all"]:>12.5f}{0.0:>12.5f}{r["delta"]:>+12.5f}'
              f'{r["full_beats_behaviour"]:>7}')
    del df; gc.collect()

json.dump(RES, open('/home/claude/final/dca_incumbent.json', 'w'), indent=2)
print('\nwrote dca_incumbent.json')
