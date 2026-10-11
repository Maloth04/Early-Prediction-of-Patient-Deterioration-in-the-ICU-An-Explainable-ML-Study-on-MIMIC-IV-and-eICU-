#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the eICU demonstration-subset feature matrix under the locked protocol
of Appendix B, with the same windowing, the same 81 predictors, the same column
names and the same conventions as engineered_features_4h_FULLSCALE.csv.

Conventions reproduced from the MIMIC-IV matrix:
  *_mean/min/max/trend  NaN when the variable is unobserved inside the window
  *_trend               OLS slope against observation time; NaN with < 2 points
  *_missing             1 when unobserved inside the window, never NaN
  *_time_since_obs_h    hours from the last observation AT OR BEFORE the window
                        end, looking back over the whole stay, NaN if none
No physiological range filtering is applied, because none was applied to the
development cohort.

Windowing (Section 3.2.2): observation window is the 24 hours ending strictly
before the prediction time; prediction times are t_k = 1440 + 240k minutes from
unit admission; tiles do not overlap; a stay exits at its first event.
"""
import numpy as np, pandas as pd, json, os

E = '/home/claude/eicu_fix/'
OUT = '/home/claude/extval/'
os.makedirs(OUT, exist_ok=True)
HRS, OBS, LANDMARK = 240, 1440, 1440          # minutes

VITALS = ['heart_rate', 'resp_rate', 'spo2', 'sbp_nbp', 'dbp_nbp', 'mbp_nbp',
          'sbp_abp', 'dbp_abp', 'mbp_abp', 'temp_c', 'temp_f']
LABS = ['lactate', 'creatinine', 'wbc']
VARS = VITALS + LABS

# ---------------------------------------------------------------- cohort ----
pat = pd.read_csv(E + 'patient.csv', low_memory=False)
pat['age_n'] = pd.to_numeric(pat.age.replace('> 89', '90'), errors='coerce')
elig = pat[(pat.age_n >= 18) & (pat.unitdischargeoffset >= OBS + HRS)]
los = dict(zip(elig.patientunitstayid, elig.unitdischargeoffset))
COHORT = set(los)
print(f'eligible stays (adult, unit stay >= {(OBS+HRS)/60:.0f} h): {len(COHORT)}')

# -------------------------------------------------------------- endpoint ----
VASO = (r'norepinephrine|epinephrine|dopamine|dobutamine|vasopressin|'
        r'phenylephrine|milrinone|levophed|neo-synephrine')
CPR = r'cardiopulmonary resuscitation|CPR|defibrillation|cardioversion'

inf = pd.read_csv(E + 'infusiondrug.csv', low_memory=False)
v = inf[inf.drugname.astype(str).str.lower().str.contains(VASO, regex=True, na=False)]
first_vaso = v.groupby('patientunitstayid').infusionoffset.min()

rc = pd.read_csv(E + 'respiratoryCare.csv', low_memory=False)
first_vent = rc.groupby('patientunitstayid').ventstartoffset.min()
first_vent = first_vent[first_vent > 0]

tr = pd.read_csv(E + 'treatment.csv', low_memory=False)
c = tr[tr.treatmentstring.astype(str).str.contains(CPR, case=False, regex=True, na=False)]
first_cpr = c.groupby('patientunitstayid').treatmentoffset.min()

first_event = (pd.concat([first_vaso.rename('t'), first_vent.rename('t'), first_cpr.rename('t')])
               .groupby(level=0).min())
first_event = first_event[first_event.index.isin(COHORT)]
EV = dict(first_event)
pre = {s for s, t in EV.items() if t <= LANDMARK}
print(f'stays with an endpoint component: {len(EV)}')
print(f'  excluded, event at or before the 24 h landmark: {len(pre)}')
MODELLED = sorted(COHORT - pre)
print(f'modelled stays: {len(MODELLED)}')

# ---------------------------------------------------------- observations ----
def long_vitals():
    cols = {'heartrate': 'heart_rate', 'respiration': 'resp_rate', 'sao2': 'spo2',
            'systemicsystolic': 'sbp_abp', 'systemicdiastolic': 'dbp_abp',
            'systemicmean': 'mbp_abp', 'temperature': 'temp_c'}
    vp = pd.read_csv(E + 'vitalPeriodic.csv', low_memory=False,
                     usecols=['patientunitstayid', 'observationoffset'] + list(cols))
    out = []
    for src, var in cols.items():
        s = vp.loc[vp[src].notna(), ['patientunitstayid', 'observationoffset', src]]
        s.columns = ['stay', 't', 'v']
        s['var'] = var
        out.append(s)
        if var == 'temp_c':
            # Appendix D: an external execution must map both temperature
            # channels to the one measurement. eICU records Celsius.
            f = s.copy(); f['v'] = s.v * 9.0 / 5.0 + 32.0; f['var'] = 'temp_f'
            out.append(f)
    acols = {'noninvasivesystolic': 'sbp_nbp', 'noninvasivediastolic': 'dbp_nbp',
             'noninvasivemean': 'mbp_nbp'}
    va = pd.read_csv(E + 'vitalAperiodic.csv', low_memory=False,
                     usecols=['patientunitstayid', 'observationoffset'] + list(acols))
    for src, var in acols.items():
        s = va.loc[va[src].notna(), ['patientunitstayid', 'observationoffset', src]]
        s.columns = ['stay', 't', 'v']
        s['var'] = var
        out.append(s)
    return pd.concat(out, ignore_index=True)


def long_labs():
    lab = pd.read_csv(E + 'lab.csv', low_memory=False,
                      usecols=['patientunitstayid', 'labresultoffset', 'labname', 'labresult'])
    want = {'lactate': 'lactate', 'creatinine': 'creatinine', 'wbc x 1000': 'wbc'}
    lab['k'] = lab.labname.astype(str).str.lower()
    s = lab[lab.k.isin(want) & lab.labresult.notna()].copy()
    s['var'] = s.k.map(want)
    s = s[['patientunitstayid', 'labresultoffset', 'labresult', 'var']]
    s.columns = ['stay', 't', 'v', 'var']
    return s


obs = pd.concat([long_vitals(), long_labs()], ignore_index=True)
obs = obs[obs.stay.isin(MODELLED)]
obs = obs.sort_values(['stay', 'var', 't'], kind='mergesort')
print(f'observations in modelled stays: {len(obs):,}')

# index: (stay, var) -> sorted (t, v) arrays
idx = {}
for (s, var), g in obs.groupby(['stay', 'var'], sort=False):
    idx[(s, var)] = (g.t.to_numpy(np.float64), g.v.to_numpy(np.float64))

# --------------------------------------------------------------- windows ----
rows = []
for s in MODELLED:
    L = los[s]
    ev = EV.get(s)
    k = 0
    while LANDMARK + (k + 1) * HRS <= L:
        t0 = LANDMARK + k * HRS                  # prediction time
        t1 = t0 + HRS                            # end of the prediction window
        lab = 1 if (ev is not None and t0 < ev <= t1) else 0
        rows.append((s, t0, lab))
        if lab:
            break                                # exit on first event
        if ev is not None and ev <= t0:
            break                                # event already happened
        k += 1
win = pd.DataFrame(rows, columns=['stay', 't0', 'label'])
print(f'instances: {len(win):,} | events: {int(win.label.sum())} '
      f'({100*win.label.mean():.2f}%) | stays: {win.stay.nunique()}')

# -------------------------------------------------------------- features ----
feat = {}
n = len(win)
for var in VARS:
    is_lab = var in LABS
    mean = np.full(n, np.nan); mn = np.full(n, np.nan); mx = np.full(n, np.nan)
    tr_ = np.full(n, np.nan); miss = np.ones(n); tso = np.full(n, np.nan)
    for i, (s, t0) in enumerate(zip(win.stay.to_numpy(), win.t0.to_numpy())):
        got = idx.get((s, var))
        if got is None:
            continue
        T, V = got
        hi = np.searchsorted(T, t0, side='left')      # strictly before t0
        if hi == 0:
            continue
        tso[i] = (t0 - T[hi - 1]) / 60.0
        lo = np.searchsorted(T, t0 - OBS, side='left')
        if hi <= lo:
            continue
        tw, vw = T[lo:hi], V[lo:hi]
        miss[i] = 0
        mean[i] = vw.mean(); mn[i] = vw.min(); mx[i] = vw.max()
        if not is_lab and len(vw) >= 2 and (tw[-1] - tw[0]) > 0:
            tr_[i] = np.polyfit(tw, vw, 1)[0]
    feat[f'{var}_mean'] = mean
    feat[f'{var}_min'] = mn
    feat[f'{var}_max'] = mx
    if not is_lab:
        feat[f'{var}_trend'] = tr_
    feat[f'{var}_missing'] = miss
    feat[f'{var}_time_since_obs_h'] = tso
    print(f'  {var:<12} observed in {100*(1-miss.mean()):5.1f}% of instances')

df = pd.DataFrame(feat)
df.insert(0, 'label', win.label.to_numpy())
df.insert(0, 'subject_id', win.stay.to_numpy())    # one unit stay = one patient here
df.insert(0, 'stay_id', win.stay.to_numpy())

# order exactly as the development matrix
ref = pd.read_csv('/mnt/user-data/uploads/outputs/cleaned_data/'
                  'engineered_features_4h_FULLSCALE.csv', nrows=1).columns.tolist()
assert set(ref) == set(df.columns), (set(ref) ^ set(df.columns))
df = df[ref]
df.to_csv(OUT + 'eicu_features_4h.csv', index=False)
json.dump({'eligible_stays': len(COHORT), 'stays_with_endpoint': len(EV),
           'excluded_pre_landmark': len(pre), 'modelled_stays': int(win.stay.nunique()),
           'instances': int(len(win)), 'events': int(win.label.sum()),
           'prevalence_pct': round(100 * float(win.label.mean()), 4)},
          open(OUT + 'eicu_cohort.json', 'w'), indent=1)
print(f'\nwritten {OUT}eicu_features_4h.csv  {df.shape}')
