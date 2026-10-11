#!/usr/bin/env python3
"""Is the disagreement result real, or an artefact of how it is constructed?

dis = logit(p_full) - logit(p_behav) is mechanically high where p_behav is low,
and "unanticipated" is defined as low p_behav. So some of the advantage is built in.

The non-circular test: WITHIN the low-concern stratum only, does the full model
discriminate events from non-events at all? If it does, there is real physiological
signal among patients nobody is watching, and targeting them is justified. If it
does not, the disagreement rule is just a relabelling of "low concern".
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

BASE = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def platt(pc, yc, pt):
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    m.fit(logit(pc).reshape(-1, 1), yc)
    return m.predict_proba(logit(pt).reshape(-1, 1))[:, 1]


df = pd.read_csv(BASE + 'engineered_features_4h_FULLSCALE.csv', low_memory=False)
allf = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
BEH = [c for c in allf if c.endswith('_missing') or c.endswith('_time_since_obs_h')]
PHYS = [c for c in allf if c not in BEH]
y = df['label'].values; g = df['subject_id'].values

res = {k: [] for k in ('full_lo', 'phys_lo', 'full_hi', 'phys_hi',
                       'n_lo', 'ev_lo', 'prev_lo', 'prev_hi')}
for rep in range(5):
    tr_all, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=1100 + rep)
                      .split(df, y, g))
    sub = df.iloc[tr_all]
    tri, cali = next(GroupShuffleSplit(1, test_size=0.25, random_state=1150 + rep)
                     .split(sub, y[tr_all], g[tr_all]))
    tr, cal = tr_all[tri], tr_all[cali]
    pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()

    P = {}
    for nm, fs in (('full', allf), ('behav', BEH), ('phys', PHYS)):
        m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=1100 + rep)
        m.fit(df.iloc[tr][fs], y[tr])
        P[nm] = platt(m.predict_proba(df.iloc[cal][fs])[:, 1], y[cal],
                      m.predict_proba(df.iloc[te][fs])[:, 1])
        del m; gc.collect()

    yt = y[te]
    cut = np.median(P['behav'])
    lo = P['behav'] < cut          # nobody is watching closely
    hi = ~lo

    for nm, mask, tag in (('full', lo, '_lo'), ('phys', lo, '_lo'),
                          ('full', hi, '_hi'), ('phys', hi, '_hi')):
        if yt[mask].sum() >= 10:
            res[nm + tag].append(roc_auc_score(yt[mask], P[nm][mask]))
    res['n_lo'].append(int(lo.sum())); res['ev_lo'].append(int(yt[lo].sum()))
    res['prev_lo'].append(float(yt[lo].mean())); res['prev_hi'].append(float(yt[hi].mean()))
    print(f'  rep {rep+1}/5', flush=True)

print('\n=== Discrimination WITHIN each concern stratum, 4-hour horizon ===')
print(f'{"Stratum":<34}{"full model":>14}{"physiology only":>18}')
for tag, nm in (('_lo', 'Low concern (nobody watching)'),
                ('_hi', 'High concern (already watched)')):
    f = np.array(res['full' + tag]); p = np.array(res['phys' + tag])
    print(f'{nm:<34}{f.mean():>9.3f} +/-{f.std():>4.3f}{p.mean():>13.3f} +/-{p.std():>4.3f}')

print(f'\n  low-concern stratum: {np.mean(res["n_lo"]):,.0f} instances, '
      f'{np.mean(res["ev_lo"]):.0f} events')
print(f'  event prevalence  low concern {100*np.mean(res["prev_lo"]):.3f}%  |  '
      f'high concern {100*np.mean(res["prev_hi"]):.3f}%')
print(f'  prevalence ratio: {np.mean(res["prev_hi"])/max(np.mean(res["prev_lo"]),1e-9):.1f}x')

print('\nVERDICT')
f_lo = np.mean(res['full_lo'])
if f_lo > 0.6:
    print(f'  The full model retains AUROC {f_lo:.3f} among patients nobody is watching.')
    print('  There IS discriminable physiological signal in that stratum, so targeting it')
    print('  is a real strategy rather than a relabelling of "low concern".')
else:
    print(f'  The full model manages only AUROC {f_lo:.3f} in that stratum, so the')
    print('  disagreement rule is close to a relabelling and should not be recommended.')

json.dump({k: [float(x) for x in v] for k, v in res.items()},
          open('/home/claude/disagree/circularity.json', 'w'), indent=2)
