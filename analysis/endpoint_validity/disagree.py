#!/usr/bin/env python3
"""Does alerting on DISAGREEMENT beat alerting on the model score?

Section 4.9 showed the model misses the deteriorations nobody anticipated, and
Section 4.8.4 showed most of its signal is the care team's own concern. That
suggests a different alerting rule: fire where the full model sees risk the
behaviour-only model does not, rather than wherever the full model scores high.

Tested at MATCHED alert budgets, so the comparison is like for like.
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
BUDGETS = [50, 100, 200, 300]          # alerts per 1000 instances


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def platt(p_cal, y_cal, p_te):
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    m.fit(logit(p_cal).reshape(-1, 1), y_cal)
    return m.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


OUT = {}
for H in ('4h', '8h'):
    df = pd.read_csv(BASE + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    allf = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    BEH = [c for c in allf if c.endswith('_missing') or c.endswith('_time_since_obs_h')]
    y = df['label'].values; g = df['subject_id'].values

    acc = {b: {k: [] for k in ('full_all', 'dis_all', 'full_un', 'dis_un',
                               'n_un', 'full_nna', 'dis_nna')} for b in BUDGETS}
    aurocs = {'full': [], 'behav': [], 'disagree': []}

    for rep in range(5):
        tr_all, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=900 + rep)
                          .split(df, y, g))
        sub = df.iloc[tr_all]
        tri, cali = next(GroupShuffleSplit(1, test_size=0.25, random_state=950 + rep)
                         .split(sub, y[tr_all], g[tr_all]))
        tr, cal = tr_all[tri], tr_all[cali]
        pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()

        P = {}
        for nm, fs in (('full', allf), ('behav', BEH)):
            m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=900 + rep)
            m.fit(df.iloc[tr][fs], y[tr])
            P[nm] = platt(m.predict_proba(df.iloc[cal][fs])[:, 1], y[cal],
                          m.predict_proba(df.iloc[te][fs])[:, 1])
            del m; gc.collect()

        yt = y[te]
        # disagreement score: how much the full model exceeds what concern alone predicts
        dis = logit(P['full']) - logit(P['behav'])

        aurocs['full'].append(roc_auc_score(yt, P['full']))
        aurocs['behav'].append(roc_auc_score(yt, P['behav']))
        aurocs['disagree'].append(roc_auc_score(yt, dis))

        # "unanticipated" = events the concern model ranks in its bottom half
        cut = np.median(P['behav'])
        unant = (yt == 1) & (P['behav'] < cut)

        for b in BUDGETS:
            frac = b / 1000.0
            tf = np.quantile(P['full'], 1 - frac)
            td = np.quantile(dis, 1 - frac)
            ff, fd = P['full'] >= tf, dis >= td
            a = acc[b]
            a['full_all'].append(int((ff & (yt == 1)).sum()))
            a['dis_all'].append(int((fd & (yt == 1)).sum()))
            a['full_un'].append(int((ff & unant).sum()))
            a['dis_un'].append(int((fd & unant).sum()))
            a['n_un'].append(int(unant.sum()))
            a['full_nna'].append(ff.sum() / max((ff & (yt == 1)).sum(), 1))
            a['dis_nna'].append(fd.sum() / max((fd & (yt == 1)).sum(), 1))
        print(f'  {H} rep {rep+1}/5', flush=True)

    print(f'\n=== {H}: discrimination of the three scores ===')
    for k in ('full', 'behav', 'disagree'):
        v = np.array(aurocs[k])
        print(f'  {k:<10} AUROC {v.mean():.3f} +/- {v.std():.3f}')

    print(f'\n=== {H}: events caught at MATCHED alert budgets (mean of 5 splits) ===')
    print(f'{"Alerts/1000":>12}{"ALL events":>24}{"UNANTICIPATED events":>26}')
    print(f'{"":>12}{"full":>10}{"disagree":>12}{"full":>12}{"disagree":>12}{"n avail":>10}')
    rows = []
    for b in BUDGETS:
        a = acc[b]
        fa, da = np.mean(a['full_all']), np.mean(a['dis_all'])
        fu, du = np.mean(a['full_un']), np.mean(a['dis_un'])
        nu = np.mean(a['n_un'])
        print(f'{b:>12}{fa:>10.0f}{da:>12.0f}{fu:>12.0f}{du:>12.0f}{nu:>10.0f}')
        rows.append(dict(budget=b, full_all=fa, dis_all=da, full_un=fu, dis_un=du,
                         n_unant=nu,
                         full_nna=float(np.mean(a['full_nna'])),
                         dis_nna=float(np.mean(a['dis_nna']))))
    OUT[H] = dict(auroc={k: float(np.mean(v)) for k, v in aurocs.items()}, budgets=rows)
    del df; gc.collect()

json.dump(OUT, open('/home/claude/disagree/disagree.json', 'w'), indent=2)
print('\nwrote disagree.json')
