#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precision calculus behind the Appendix B.1 event-count precondition.

Appendix B.1 requires at least 150 post-landmark events but shows no arithmetic,
so the bar reads as arbitrary and the thesis rests a decision not to execute on
an unexplained number. This computes what the registered criteria actually
require, using only quantities already in the thesis.

Inputs, all registered or reported:
  internal benchmark          0.769 AUROC, Table 4.41, four hours
  internal split              401 events in 52,279 instances, Table 4.41 caption
  registered tolerated fall   0.05 AUROC, Section 3.11 / Appendix B.4
  external event:non-event    27 : 17,749 as observed in the demonstration release
Variance of AUROC by Hanley and McNeil (1982).

Validation of the analytic variance against the thesis's own measurement is
printed first: the bootstrap interval reported in Section 4.12.1 is 0.207 wide
at 27 events, so if the formula does not reproduce that it cannot be trusted
prospectively.
"""
import numpy as np, json
from scipy.stats import norm

def hm_var(auc, npos, nneg):
    q1 = auc / (2 - auc); q2 = 2 * auc ** 2 / (1 + auc)
    return (auc*(1-auc) + (npos-1)*(q1-auc**2) + (nneg-1)*(q2-auc**2)) / (npos*nneg)

AUC_INT, DROP = 0.769, 0.05
AUC_EXT = AUC_INT - DROP
INT_POS, INT_INST = 401, 52279                 # Table 4.41, four hours
INT_NEG = INT_INST - INT_POS
R_EICU = (17776 - 27) / 27                     # observed external ratio, 657.4 : 1
R_MIMIC = (263675 - 2080) / 2080               # MIMIC-IV four-hour ratio, 125.8 : 1
Z80 = norm.ppf(0.975) + norm.ppf(0.80)
Z90 = norm.ppf(0.975) + norm.ppf(0.90)
v_int = hm_var(AUC_INT, INT_POS, INT_NEG)
out = {'internal_se': round(float(np.sqrt(v_int)), 4),
       'ratio_eicu': round(R_EICU, 1), 'ratio_mimic': round(R_MIMIC, 1)}

# --- 0. does the analytic variance reproduce the measured bootstrap width? ---
w27 = 2 * 1.96 * np.sqrt(hm_var(0.749, 27, 27 * R_EICU))
out['analytic_width_at_27'] = round(float(w27), 3)
print(f'analytic 95% width at 27 events, AUROC 0.749: {w27:.3f}')
print(f'measured bootstrap width, Section 4.12.1:     0.207   -> difference {abs(w27-0.207):.3f}\n')

# --- 1. events required by the registered criterion ---
def need(ratio, z, two_sample):
    lo, hi = 10, 500000
    while lo < hi:
        m = (lo + hi) // 2
        se = np.sqrt((v_int if two_sample else 0) + hm_var(AUC_EXT, m, m*ratio))
        if DROP / se >= z: hi = m
        else: lo = m + 1
    return lo

for rl, ratio in (('eicu', R_EICU), ('mimic_prevalence', R_MIMIC)):
    for pl, z in (('80', Z80), ('90', Z90)):
        for sl, ts in (('two_sample', True), ('one_sample', False)):
            n = need(ratio, z, ts)
            out[f'events_{rl}_{pl}pct_{sl}'] = n
            if rl == 'eicu':
                print(f'  {pl}% power, {sl.replace("_"," ")}: {n:,} events')
print()

# --- 2. what each event count can resolve ---
for n in (27, 29, 48, 150, 266, 642):
    se2 = np.sqrt(v_int + hm_var(AUC_EXT, n, n*R_EICU))
    se1 = np.sqrt(hm_var(AUC_EXT, n, n*R_EICU))
    row = {'ci_width': round(float(2*1.96*se1), 3),
           'mde_two_sample': round(float(Z80*se2), 3),
           'mde_one_sample': round(float(Z80*se1), 3)}
    out[f'n{n}'] = row
    print(f'  {n:>5} events: CI width {row["ci_width"]:.3f}, smallest detectable fall '
          f'{row["mde_two_sample"]:.3f} (two-sample) / {row["mde_one_sample"]:.3f} (one-sample)')

# --- 3. the floor the internal split imposes ---
floor = Z80 * np.sqrt(v_int)
lo, hi = 10, 500000
while lo < hi:
    m = (lo + hi) // 2
    if hm_var(AUC_EXT, m, m*R_EICU) <= v_int: hi = m
    else: lo = m + 1
out['internal_floor_mde'] = round(float(floor), 3)
out['events_where_variances_equal'] = lo
print(f'\ninternal split alone bounds the two-sample test at {floor:.3f}; no external')
print(f'sample size goes below it. External variance equals internal at {lo:,} events.')

# --- 4. the full release, projected ---
FULL = 27 * (200859 / 2520)
se2 = np.sqrt(v_int + hm_var(AUC_EXT, FULL, FULL*R_EICU))
se1 = np.sqrt(hm_var(AUC_EXT, FULL, FULL*R_EICU))
out['full_release'] = {'projected_events': int(round(FULL)),
                       'ci_width': round(float(2*1.96*se1), 3),
                       'mde_two_sample': round(float(Z80*se2), 3),
                       'mde_one_sample': round(float(Z80*se1), 3)}
print(f'\nfull release, {int(round(FULL)):,} projected events: CI width '
      f'{2*1.96*se1:.3f}, smallest detectable fall {Z80*se2:.3f} / {Z80*se1:.3f}')

# --- 5. what 27 events do establish ---
se27 = np.sqrt(hm_var(0.749, 27, 27*R_EICU))
z = (0.749 - 0.5) / se27
out['above_chance_4h'] = {'auroc': 0.749, 'z': round(float(z), 2),
                          'p': float(2*(1-norm.cdf(z)))}
se27b = np.sqrt(hm_var(0.683, 27, 27*(8581-27)/27))
zb = (0.683 - 0.5) / se27b
out['above_chance_8h'] = {'auroc': 0.683, 'z': round(float(zb), 2),
                          'p': float(2*(1-norm.cdf(zb)))}
print(f'\n27 events vs chance: 4 h z = {z:.2f}, p = {2*(1-norm.cdf(z)):.1e}; '
      f'8 h z = {zb:.2f}, p = {2*(1-norm.cdf(zb)):.1e}')
json.dump(out, open('/home/claude/extval/precision.json', 'w'), indent=1)
print('\nwritten precision.json')
