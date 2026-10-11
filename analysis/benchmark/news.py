#!/usr/bin/env python3
"""Does the model beat the early warning score clinicians already use?

Chapter 2 frames track-and-trigger scores as the clinical comparator but never
computes one. NEWS2 is reconstructed here from the worst observation in each
window - which is how a bedside team would apply it - using the five of seven
components the cohort supports. Supplemental oxygen and level of consciousness
are not extracted in the present pipeline, so the score is a lower bound on what
a complete NEWS2 would achieve, and that is stated rather than glossed.
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


def band(x, cuts, pts):
    """Score by band; NaN -> 0 (absent observation cannot trigger)."""
    out = np.zeros(len(x))
    v = x.values if hasattr(x, 'values') else x
    for (lo, hi), p in zip(cuts, pts):
        out = np.where((v >= lo) & (v < hi), p, out)
    return np.nan_to_num(out)


def news2(d):
    """NEWS2, five components, computed on the worst value in the window."""
    s = np.zeros(len(d))
    # respiratory rate (worst = highest, and lowest for bradypnoea)
    rr = d['resp_rate_max']
    s += band(rr, [(-1, 9), (9, 12), (12, 21), (21, 25), (25, 1e9)], [3, 1, 0, 2, 3])
    rrl = d['resp_rate_min']
    s += np.where(rrl.fillna(99) <= 8, 3, 0)
    # SpO2 (worst = lowest)
    sp = d['spo2_min']
    s += band(sp, [(-1, 92), (92, 94), (94, 96), (96, 1e9)], [3, 2, 1, 0])
    # systolic BP (worst = lowest; very high also scores)
    sb = d['sbp_nbp_min']
    s += band(sb, [(-1, 91), (91, 101), (101, 111), (111, 220), (220, 1e9)], [3, 2, 1, 0, 3])
    # pulse
    hr = d['heart_rate_max']
    s += band(hr, [(-1, 41), (41, 51), (51, 91), (91, 111), (111, 131), (131, 1e9)],
              [3, 1, 0, 1, 2, 3])
    hrl = d['heart_rate_min']
    s += np.where(hrl.fillna(99) <= 40, 3, 0)
    # temperature
    t = d['temp_c_mean'].copy()
    tf = d['temp_f_mean']
    t = t.fillna((tf - 32) * 5 / 9)
    s += band(t, [(-1, 35.01), (35.01, 36.01), (36.01, 38.01), (38.01, 39.01), (39.01, 1e9)],
              [3, 1, 0, 1, 2])
    return s


OUT = {}
for H in ('4h', '8h'):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    PHYS = [c for c in feats if not (c.endswith('_missing') or c.endswith('_time_since_obs_h'))]
    y = df['label'].values; g = df['subject_id'].values
    df['NEWS'] = news2(df)
    if H == '4h':
        print(f'NEWS distribution (4h): median {df.NEWS.median():.0f}  '
              f'mean {df.NEWS.mean():.2f}  max {df.NEWS.max():.0f}')
        print('  proportion at each clinical trigger level:')
        for thr, lab in ((5, 'urgent response (>=5)'), (7, 'emergency response (>=7)')):
            m = df.NEWS >= thr
            print(f'    {lab:<28} {100*m.mean():>5.1f}% of windows,  '
                  f'event rate {100*y[m.values].mean():.2f}% vs {100*y.mean():.2f}% overall')

    res = {k: [] for k in ('news', 'phys', 'full', 'news_ap', 'full_ap', 'inc')}
    op = []
    for rep in range(5):
        tr_all, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=1300 + rep)
                          .split(df, y, g))
        sub = df.iloc[tr_all]
        tri, cali = next(GroupShuffleSplit(1, test_size=0.25, random_state=1350 + rep)
                         .split(sub, y[tr_all], g[tr_all]))
        tr, cal = tr_all[tri], tr_all[cali]
        pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
        yt = y[te]

        res['news'].append(roc_auc_score(yt, df.iloc[te]['NEWS']))
        res['news_ap'].append(average_precision_score(yt, df.iloc[te]['NEWS']))
        P = {}
        for nm, fs in (('phys', PHYS), ('full', feats)):
            m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=1300 + rep)
            m.fit(df.iloc[tr][fs], y[tr])
            P[nm] = m.predict_proba(df.iloc[te][fs])[:, 1]
            res[nm].append(roc_auc_score(yt, P[nm]))
            del m; gc.collect()
        res['full_ap'].append(average_precision_score(yt, P['full']))
        res['inc'].append(res['full'][-1] - res['news'][-1])

        if rep == 0 and H == '4h':
            nv = df.iloc[te]['NEWS'].values
            for thr in (5, 7):
                flag = nv >= thr
                sens = (flag & (yt == 1)).sum() / max((yt == 1).sum(), 1)
                # model at the SAME alert rate
                tm = np.quantile(P['full'], 1 - flag.mean())
                fm = P['full'] >= tm
                sm = (fm & (yt == 1)).sum() / max((yt == 1).sum(), 1)
                op.append(dict(threshold=thr, alert_rate=float(flag.mean()),
                               news_sens=float(sens), model_sens=float(sm),
                               news_ppv=float(yt[flag].mean()) if flag.sum() else 0.0,
                               model_ppv=float(yt[fm].mean()) if fm.sum() else 0.0))
        print(f'  {H} rep {rep+1}/5', flush=True)

    print(f'\n=== {H}: discrimination against the clinical benchmark ===')
    for k, lab in (('news', 'NEWS2 (5 components)'), ('phys', 'Physiology-only model'),
                   ('full', 'Full model')):
        v = np.array(res[k])
        print(f'  {lab:<26} AUROC {v.mean():.3f} +/- {v.std():.3f}')
    inc = np.array(res['inc'])
    print(f'  Incremental value of the model over NEWS2: '
          f'{inc.mean():+.3f} (SD {inc.std():.3f}, {int((inc>0).sum())}/5 splits positive)')
    print(f'  AUPRC  NEWS2 {np.mean(res["news_ap"]):.4f}   full model '
          f'{np.mean(res["full_ap"]):.4f}')
    if op:
        print(f'\n=== {H}: at the standard NEWS2 trigger thresholds ===')
        for o in op:
            print(f'  NEWS >= {o["threshold"]}: fires on {100*o["alert_rate"]:.1f}% of windows | '
                  f'sensitivity NEWS {100*o["news_sens"]:.1f}% vs model '
                  f'{100*o["model_sens"]:.1f}% at the same alert rate | '
                  f'PPV {100*o["news_ppv"]:.2f}% vs {100*o["model_ppv"]:.2f}%')
    OUT[H] = dict(auroc={k: float(np.mean(res[k])) for k in ('news', 'phys', 'full')},
                  inc=float(inc.mean()), inc_sd=float(inc.std()),
                  auprc_news=float(np.mean(res['news_ap'])),
                  auprc_full=float(np.mean(res['full_ap'])), operating=op)
    del df; gc.collect()

json.dump(OUT, open('/home/claude/news/news.json', 'w'), indent=2)
print('\nwrote news.json')
