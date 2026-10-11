#!/usr/bin/env python3
"""Independent check of the Hanley-McNeil implementation used for Table B.2.

The analytic variance is already checked against the thesis's own bootstrap
(0.214 against 0.207 at 27 events). This checks the implementation itself by
Monte Carlo at sizes small enough to simulate, under a binormal model whose
population AUROC is set exactly, and across three event-to-non-event ratios.
If the formula is coded correctly the empirical SD matches sqrt(hm_var).
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

def hm_var(auc, npos, nneg):
    q1 = auc/(2-auc); q2 = 2*auc**2/(1+auc)
    return (auc*(1-auc)+(npos-1)*(q1-auc**2)+(nneg-1)*(q2-auc**2))/(npos*nneg)

rng = np.random.default_rng(7)
for AUC in (0.719, 0.769):
    mu = norm.ppf(AUC) * np.sqrt(2)           # binormal, equal variance
    for npos, nneg in ((150, 2000), (27, 1800), (150, 9000)):
        est = []
        for _ in range(3000):
            s = np.concatenate([rng.normal(mu, 1, npos), rng.normal(0, 1, nneg)])
            y = np.concatenate([np.ones(npos), np.zeros(nneg)])
            est.append(roc_auc_score(y, s))
        emp, ana = np.std(est, ddof=1), np.sqrt(hm_var(AUC, npos, nneg))
        print(f'AUC {AUC}  n+={npos:>4} n-={nneg:>5}  '
              f'empirical SD {emp:.5f}  Hanley-McNeil {ana:.5f}  ratio {emp/ana:.3f}  '
              f'(mean AUROC {np.mean(est):.4f})')

# ---- the bracket: HM as upper bound, binormal-scaled as lower bound ----
print()
K_EXT, K_INT = 0.881**2, 0.836**2          # var(binormal)/var(HM), from above
A, D = 0.769, 0.05; AE = A - D
R = (17776-27)/27
Z = norm.ppf(0.975) + norm.ppf(0.80)
vi_hm = hm_var(A, 401, 52279-401)
def req(k_ext, k_int, two_sample):
    lo, hi = 10, 500000
    while lo < hi:
        m = (lo+hi)//2
        se = np.sqrt((vi_hm*k_int if two_sample else 0) + hm_var(AE, m, m*R)*k_ext)
        if D/se >= Z: hi = m
        else: lo = m+1
    return lo
for lbl, ke, ki in (('Hanley-McNeil', 1.0, 1.0), ('binormal-scaled', K_EXT, K_INT)):
    print(f'{lbl:>16}: 80% power needs {req(ke,ki,False):>5,} events (one-sample), '
          f'{req(ke,ki,True):>6,} (two-sample)')
for n in (150, 2152):
    for lbl, ke, ki in (('Hanley-McNeil', 1.0, 1.0), ('binormal-scaled', K_EXT, K_INT)):
        s1 = np.sqrt(hm_var(AE, n, n*R)*ke)
        s2 = np.sqrt(vi_hm*ki + hm_var(AE, n, n*R)*ke)
        print(f'  n={n:>5} {lbl:>16}: MDE {Z*s1:.3f} (one-sample) / {Z*s2:.3f} (two-sample)')
