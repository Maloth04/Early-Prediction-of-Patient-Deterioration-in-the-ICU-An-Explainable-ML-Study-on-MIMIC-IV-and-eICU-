#!/usr/bin/env python3
"""v6 Section 6.3 sensitivity analyses + the calibration reporting TRIPOD+AI requires."""
import pandas as pd, numpy as np, gc, json, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

BASE = '/mnt/user-data/uploads/outputs/cleaned_data/'
OUT = {}


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration(y, p):
    """Calibration slope, intercept (calibration-in-the-large), Brier."""
    lp = logit(p).reshape(-1, 1)
    slope = float(LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
                  .fit(lp, y).coef_[0][0])
    # intercept with slope fixed at 1 -> fit intercept-only on offset
    off = logit(p)
    lo, hi = -20.0, 20.0
    for _ in range(80):                      # bisection on the score equation
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(off + mid)))).mean() < y.mean():
            lo = mid
        else:
            hi = mid
    intercept = float((lo + hi) / 2)
    return slope, intercept, float(brier_score_loss(y, p))


def platt(p_tr, y_tr, p_te):
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    m.fit(logit(p_tr).reshape(-1, 1), y_tr)
    return m.predict_proba(logit(p_te).reshape(-1, 1))[:, 1]


PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
              tree_method='hist', n_jobs=4, verbosity=0, random_state=42)


def fit_eval(Xtr, ytr, Xte, yte, Xcal=None, ycal=None, pos_weight=None):
    m = xgb.XGBClassifier(**PARAMS, scale_pos_weight=pos_weight) if pos_weight \
        else xgb.XGBClassifier(**PARAMS)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    r = dict(auroc=float(roc_auc_score(yte, p)), auprc=float(average_precision_score(yte, p)))
    s, i, b = calibration(yte, p)
    r.update(slope_raw=s, intercept_raw=i, brier_raw=b)
    if Xcal is not None:
        pc_tr = m.predict_proba(Xcal)[:, 1]
        pr = platt(pc_tr, ycal, p)
        s2, i2, b2 = calibration(yte, pr)
        r.update(slope_cal=s2, intercept_cal=i2, brier_cal=b2,
                 auroc_cal=float(roc_auc_score(yte, pr)))
    del m; gc.collect()
    return r


def load(h):
    df = pd.read_csv(BASE + f'engineered_features_{h}_FULLSCALE.csv', low_memory=False)
    feats = [c for c in df.columns if c not in ('stay_id', 'subject_id', 'label')]
    return df, feats


def three_way(df, feats, seed=42):
    """patient-level train / calibration / test"""
    y = df['label'].values; g = df['subject_id'].values
    tr_all, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(df, y, g))
    sub = df.iloc[tr_all]
    tr_i, cal_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed + 1)
                       .split(sub, y[tr_all], g[tr_all]))
    return tr_all[tr_i], tr_all[cal_i], te


# ============ 1. PRIMARY + CALIBRATION, BOTH HORIZONS ============
print('=== 1. Primary model: discrimination and calibration ===')
for h in ('4h', '8h'):
    df, feats = load(h)
    y = df['label'].values
    tr, cal, te = three_way(df, feats)
    pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()
    r = fit_eval(df.iloc[tr][feats], y[tr], df.iloc[te][feats], y[te],
                 df.iloc[cal][feats], y[cal], pos_weight=pw)
    OUT[f'primary_{h}'] = r
    print(f'  {h}: AUROC {r["auroc"]:.3f} AUPRC {r["auprc"]:.4f}')
    print(f'      RAW       slope {r["slope_raw"]:6.3f}  intercept {r["intercept_raw"]:7.3f}  Brier {r["brier_raw"]:.5f}')
    print(f'      PLATT     slope {r["slope_cal"]:6.3f}  intercept {r["intercept_cal"]:7.3f}  Brier {r["brier_cal"]:.5f}')
    del df; gc.collect()

# ============ 2. CLASS-IMBALANCE HANDLING (4h) ============
print('\n=== 2. Class-imbalance handling (4h) ===')
df, feats = load('4h'); y = df['label'].values
tr, cal, te = three_way(df, feats)
pw = (len(y[tr]) - y[tr].sum()) / y[tr].sum()

variants = {}
variants['class weighting (primary)'] = fit_eval(
    df.iloc[tr][feats], y[tr], df.iloc[te][feats], y[te],
    df.iloc[cal][feats], y[cal], pos_weight=pw)
variants['no weighting'] = fit_eval(
    df.iloc[tr][feats], y[tr], df.iloc[te][feats], y[te],
    df.iloc[cal][feats], y[cal])
# random undersampling of the majority class within train only
rng = np.random.RandomState(42)
pos_idx = tr[y[tr] == 1]
neg_idx = tr[y[tr] == 0]
keep = rng.choice(neg_idx, size=min(len(neg_idx), 20 * len(pos_idx)), replace=False)
us = np.concatenate([pos_idx, keep])
variants['undersampling 20:1'] = fit_eval(
    df.iloc[us][feats], y[us], df.iloc[te][feats], y[te],
    df.iloc[cal][feats], y[cal])
OUT['imbalance'] = variants
for kk, v in variants.items():
    print(f'  {kk:<26} AUROC {v["auroc"]:.3f}  AUPRC {v["auprc"]:.4f}  slope(raw) {v["slope_raw"]:.3f}')

# ============ 3. IMPUTATION STRATEGY (4h) ============
print('\n=== 3. Imputation strategy (4h) ===')
imp = {}
imp['native XGBoost (primary)'] = variants['class weighting (primary)']
Xm = df[feats].copy()
means = Xm.iloc[tr].mean()
Xm = Xm.fillna(means)
imp['mean imputation'] = fit_eval(Xm.iloc[tr], y[tr], Xm.iloc[te], y[te],
                                  Xm.iloc[cal], y[cal], pos_weight=pw)
Xz = df[feats].fillna(0)
imp['zero fill'] = fit_eval(Xz.iloc[tr], y[tr], Xz.iloc[te], y[te],
                            Xz.iloc[cal], y[cal], pos_weight=pw)
OUT['imputation'] = imp
for kk, v in imp.items():
    print(f'  {kk:<26} AUROC {v["auroc"]:.3f}  AUPRC {v["auprc"]:.4f}')
del Xm, Xz; gc.collect()

# ============ 4. MINIMUM STAY-LENGTH THRESHOLD (4h) ============
print('\n=== 4. Minimum ICU stay length (4h) ===')
icu = pd.read_csv(BASE + 'icustays_CLEANED.csv', low_memory=False)
loscol = 'los' if 'los' in icu.columns else [c for c in icu.columns if 'los' in c.lower()][0]
los = icu.set_index('stay_id')[loscol]          # days
df['_los_h'] = df['stay_id'].map(los) * 24.0
print(f'  LOS column used: {loscol}; median {df["_los_h"].median():.1f} h')
stay = {}
for thr in (24, 36, 48, 72):
    mask = df['_los_h'] >= thr
    sub = df[mask]
    if sub['label'].sum() < 100:
        print(f'  >= {thr}h: only {int(sub["label"].sum())} events, skipped'); continue
    ys = sub['label'].values; gs = sub['subject_id'].values
    a, b, c = three_way(sub.reset_index(drop=True), feats)
    s2 = sub.reset_index(drop=True)
    pw2 = (len(ys[a]) - ys[a].sum()) / ys[a].sum()
    r = fit_eval(s2.iloc[a][feats], ys[a], s2.iloc[c][feats], ys[c],
                 s2.iloc[b][feats], ys[b], pos_weight=pw2)
    r['n_instances'] = int(len(sub)); r['n_events'] = int(sub['label'].sum())
    stay[f'>= {thr} h'] = r
    print(f'  >= {thr:>2}h: n={len(sub):>7,} events={int(sub["label"].sum()):>5,}  '
          f'AUROC {r["auroc"]:.3f}  AUPRC {r["auprc"]:.4f}')
OUT['stay_length'] = stay

json.dump(OUT, open('/home/claude/sens/sensitivity_results.json', 'w'), indent=2)
print('\nwrote sensitivity_results.json')
