#!/usr/bin/env python3
"""Does the headline finding survive correcting the endpoint?

Section 4.10.1 showed the endpoint codes 1,493 deaths as non-events, because it
records an escalation of treatment rather than a patient state. That is also the
strongest objection to the decomposition result: if measurement behaviour predicts
escalation largely because escalation is a CLINICIAN DECISION, then under an
endpoint that includes death - which is not a decision - the behaviour block's
advantage should shrink or vanish.

This reconstructs window timing, builds the corrected endpoint, and re-runs the
decomposition. It is a direct test of the author's own principal finding.
"""
import pandas as pd, numpy as np, json, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
PAR = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5,
           subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
           tree_method='hist', n_jobs=4, verbosity=0)
OUT = {}

for H, HRS in (('4h', 4), ('8h', 8)):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    icu = pd.read_csv(B + 'icustays_CLEANED.csv',
                      usecols=['stay_id', 'hadm_id', 'intime', 'outtime'], low_memory=False)
    adm = pd.read_csv('/mnt/user-data/uploads/admissions.csv',
                      usecols=['hadm_id', 'deathtime'], low_memory=False)
    icu = icu.merge(adm, on='hadm_id', how='left')
    for c in ('intime', 'outtime', 'deathtime'):
        icu[c] = pd.to_datetime(icu[c], errors='coerce')

    # ---- reconstruct the landmark time of every window ----
    df['k'] = df.groupby('stay_id').cumcount()
    df = df.merge(icu[['stay_id', 'intime', 'outtime', 'deathtime']], on='stay_id', how='left')
    df['landmark'] = df['intime'] + pd.to_timedelta(24 + df['k'] * HRS, unit='h')
    df['horizon_end'] = df['landmark'] + pd.to_timedelta(HRS, unit='h')

    # sanity: landmarks must lie inside the stay
    inside = (df.landmark >= df.intime) & (df.landmark <= df.outtime + pd.Timedelta('1h'))
    print(f'[{H}] windows {len(df):,}  landmark inside stay: {100*inside.mean():.1f}%')

    # ---- corrected endpoint: escalation OR death within the horizon ----
    death_in = (df.deathtime.notna() &
                (df.deathtime > df.landmark) & (df.deathtime <= df.horizon_end))
    df['label_corr'] = ((df.label == 1) | death_in).astype(int)
    n_add = int((df.label_corr == 1).sum() - (df.label == 1).sum())
    print(f'  original events {int(df.label.sum()):,}  '
          f'corrected {int(df.label_corr.sum()):,}  (+{n_add:,} deaths added, '
          f'{100*n_add/max(int(df.label.sum()),1):.1f}% increase)')
    do = int((death_in & (df.label == 0)).sum())
    print(f'  deaths in horizon that were previously NON-events: {do:,}')

    feats = [c for c in df.columns if c not in
             ('stay_id', 'subject_id', 'label', 'label_corr', 'k', 'intime',
              'outtime', 'deathtime', 'landmark', 'horizon_end')]
    BEH = [c for c in feats if c.endswith('_missing') or c.endswith('_time_since_obs_h')]
    PHYS = [c for c in feats if c not in BEH]
    g = df['subject_id'].values

    for tag, yv in (('original', df.label.values), ('corrected', df.label_corr.values)):
        acc = {k: [] for k in ('PHYS', 'BEHAV', 'FULL', 'PHYS_ap', 'BEHAV_ap', 'FULL_ap')}
        for rep in range(3):
            tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=1500 + rep)
                          .split(df, yv, g))
            pw = (len(yv[tr]) - yv[tr].sum()) / max(yv[tr].sum(), 1)
            for nm, fs in (('PHYS', PHYS), ('BEHAV', BEH), ('FULL', feats)):
                m = xgb.XGBClassifier(**PAR, scale_pos_weight=pw, random_state=1500 + rep)
                m.fit(df.iloc[tr][fs], yv[tr])
                p = m.predict_proba(df.iloc[te][fs])[:, 1]
                acc[nm].append(roc_auc_score(yv[te], p))
                acc[nm + '_ap'].append(average_precision_score(yv[te], p))
                del m; gc.collect()
        ph, be, fu = (np.mean(acc['PHYS']), np.mean(acc['BEHAV']), np.mean(acc['FULL']))
        share = 100 * (be - 0.5) / (fu - 0.5)
        OUT[f'{H}_{tag}'] = dict(phys=float(ph), behav=float(be), full=float(fu),
                                 share=float(share),
                                 phys_ap=float(np.mean(acc['PHYS_ap'])),
                                 behav_ap=float(np.mean(acc['BEHAV_ap'])),
                                 full_ap=float(np.mean(acc['FULL_ap'])),
                                 n_pos=int(yv.sum()))
        print(f'  [{tag:<9}] PHYS {ph:.3f}  BEHAV {be:.3f}  FULL {fu:.3f}  '
              f'| behaviour share {share:.1f}%  | AUPRC full {np.mean(acc["FULL_ap"]):.4f}')
    del df; gc.collect()
    print()

print('=' * 76)
print('DOES THE FINDING SURVIVE?')
print('=' * 76)
for H in ('4h', '8h'):
    o, c = OUT[f'{H}_original'], OUT[f'{H}_corrected']
    print(f'{H}: behaviour share {o["share"]:.1f}% -> {c["share"]:.1f}%  '
          f'({c["share"]-o["share"]:+.1f} pts)')
    print(f'    BEHAV minus PHYS: {o["behav"]-o["phys"]:+.3f} -> {c["behav"]-c["phys"]:+.3f}')
    print(f'    events {o["n_pos"]:,} -> {c["n_pos"]:,}')
json.dump(OUT, open('/home/claude/endpoint/corrected.json', 'w'), indent=2)
print('\nwrote corrected.json')
