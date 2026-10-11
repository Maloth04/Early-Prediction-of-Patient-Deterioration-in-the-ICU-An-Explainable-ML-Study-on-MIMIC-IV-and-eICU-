#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Figures 4.1, 4.2 and 4.3 from the thesis tables.

Every number below is taken from the table it belongs to, so the figures can no
longer disagree with the text:
  Fig 4.1A  Table 4.26 + the NEWS2 reconstruction (Section 4.3.5)
  Fig 4.1B  computed here from the same NEWS2 scoring code as Section 4.3.5
  Fig 4.1C  Section 5.4 increments, relabelled: the behaviour-only comparator is
            the observation record, NOT the care team's judgement (Section 5.4
            says so explicitly), which is what the old panel claimed
  Fig 4.2   Table 4.33, and the dotted reference is 129 per split (Section 4.9.1)
  Fig 4.3A  Table 4.34     Fig 4.3B  Table 4.35     Fig 4.3C  Table 4.36
The palette is sampled from the existing figures so the regenerated ones stay
consistent with Figure 4.4, which is unchanged.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
OUT = '/tmp/claude-0/figs/'
NAVY, TEAL, ORANGE, GREEN, RED, GREY = '#0e3a52', '#0e7c74', '#a8660d', '#1e8f5f', '#a03a2b', '#688392'
LIGHT = '#d3dce0'
plt.rcParams.update({'font.size': 15, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#444', 'figure.dpi': 110})


def titled(ax, letter, text):
    ax.set_title(f'{letter}  {text}', loc='left', fontsize=17, fontweight='bold',
                 color=NAVY, pad=14)


# ------------------------------------------------- NEWS2 distribution ----
import importlib.util as _il
_spec = _il.spec_from_file_location('newsmod', '/home/claude/news/_fn.py')
_m = _il.module_from_spec(_spec)
_m.np = np                   # _fn.py relies on its caller's numpy import
_spec.loader.exec_module(_m)
news2 = _m.news2            # the exact scoring used in Section 4.3.5


cols = ['resp_rate_max', 'resp_rate_min', 'spo2_min', 'sbp_nbp_min', 'heart_rate_max',
        'heart_rate_min', 'temp_c_mean', 'temp_f_mean']
dev = pd.read_csv(B + 'engineered_features_4h_FULLSCALE.csv', usecols=cols, low_memory=False)
sc = news2(dev)
print(f'NEWS2: median {np.median(sc):.0f} | >=5 {100*(sc>=5).mean():.1f}% | >=7 {100*(sc>=7).mean():.1f}%')
hist = pd.Series(sc).value_counts(normalize=True).sort_index() * 100

# ======================================================== Figure 4.1 ====
fig, ax = plt.subplots(1, 3, figsize=(2941/110, 704/110))
lbl = ['NEWS2\n(5 comp.)', 'Physiology\nmodel', 'Measurement\nbehaviour', 'Full\nmodel']
h4 = [0.597, 0.711, 0.723, 0.783]; h8 = [0.595, 0.685, 0.708, 0.759]
x = np.arange(4); w = 0.38
ax[0].bar(x - w/2, h4, w, color=TEAL, label='4-hour')
ax[0].bar(x + w/2, h8, w, color=NAVY, label='8-hour')
for i, (a, b) in enumerate(zip(h4, h8)):
    ax[0].text(i - w/2, a + .004, f'{a:.3f}', ha='center', fontsize=13.5, fontweight='bold', color=NAVY)
    ax[0].text(i + w/2, b + .004, f'{b:.3f}', ha='center', fontsize=13.5, color=NAVY)
ax[0].set_xticks(x); ax[0].set_xticklabels(lbl); ax[0].set_ylim(0.5, 0.81)
ax[0].set_ylabel('AUROC'); ax[0].legend(frameon=False, loc='upper left')
titled(ax[0], 'A', 'The formal score is the weakest baseline')

cl = [LIGHT if s < 5 else (ORANGE if s < 7 else RED) for s in hist.index]
ax[1].bar(hist.index, hist.values, color=cl, width=.85)
ax[1].set_xlim(-1, 16.5); ax[1].set_ylim(0, hist.max() * 1.42)
ax[1].axvline(4.5, ls='--', color=ORANGE, lw=2); ax[1].axvline(6.5, ls='--', color=RED, lw=2)
ax[1].annotate(f'NEWS \u2265 5  \u2192  {100*(sc>=5).mean():.1f}% of windows',
               xy=(4.5, hist.max()*1.02), xytext=(7.6, hist.max()*1.30), color=ORANGE,
               fontsize=13.5, arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.6))
ax[1].annotate(f'NEWS \u2265 7  \u2192  {100*(sc>=7).mean():.1f}% of windows',
               xy=(6.5, hist.max()*0.92), xytext=(9.4, hist.max()*1.14), color=RED,
               fontsize=13.5, arrowprops=dict(arrowstyle='->', color=RED, lw=1.6))
ax[1].text(-0.8, hist.max()*1.30, f'median score {np.median(sc):.0f}', color=GREY,
           fontsize=13.5, style='italic')
ax[1].set_xlabel('NEWS2 score (five components)'); ax[1].set_ylabel('% of modelled windows')
titled(ax[1], 'B', 'NEWS2 saturates in the ICU')

inc = [0.186, 0.060]
names = ['over NEWS2\n(the formal score)', 'over the behaviour-only\ncomparator (the observation\nrecord)']
ax[2].barh([1, 0], inc, color=[GREEN, ORANGE], height=.45)
for i, v in zip([1, 0], inc):
    ax[2].text(v + .004, i, f'+{v:.3f}', va='center', fontsize=17, fontweight='bold', color=NAVY)
ax[2].set_yticks([1, 0]); ax[2].set_yticklabels(names, fontsize=14)
ax[2].set_xlim(0, 0.205); ax[2].set_xlabel('Incremental AUROC, 4-hour horizon')
ax[2].text(0.0, -0.62, 'The record of how the team observed the patient is a far stronger\n'
                       'baseline than the instrument built to support them. It is not the\n'
                       'team’s judgement itself (Section 5.4).',
           fontsize=13, style='italic', color=GREY, va='top')
titled(ax[2], 'C', 'Which baseline you choose decides the story')
fig.tight_layout(); fig.savefig(OUT + 'new3.png', bbox_inches='tight'); plt.close(fig)

# ======================================================== Figure 4.2 ====
fig, ax = plt.subplots(1, 3, figsize=(2861/110, 709/110))
bud = [50, 100, 200, 300]
dis_u = [25, 39, 64, 79]; mod_u = [7, 14, 31, 47]
dis_a = [47, 91, 173, 236]; mod_a = [171, 227, 303, 357]
ax[0].axhline(129, ls=':', color=GREY, lw=2)
ax[0].text(300, 132, 'all 129 available', ha='right', fontsize=13.5, color=GREY, style='italic')
ax[0].plot(bud, dis_u, '-o', color=TEAL, lw=3, ms=11, label='Alert on disagreement')
ax[0].plot(bud, mod_u, '--s', color=GREY, lw=3, ms=10, label='Alert on model score')
for b, v in zip(bud, dis_u):
    ax[0].text(b, v + 4, str(v), ha='center', fontsize=14, fontweight='bold', color=TEAL)
for b, v in zip(bud, mod_u):
    ax[0].text(b, v - 10 if v > 12 else v + 5, str(v), ha='center', fontsize=14, color=GREY)
ax[0].set_ylim(0, 172); ax[0].set_xticks(bud)
ax[0].set_xlabel('Alert budget (alerts per 1,000 instances)')
ax[0].set_ylabel('Unanticipated events caught'); ax[0].legend(frameon=False, loc='upper left')
titled(ax[0], 'A', 'The deteriorations nobody saw coming')

x = np.arange(4); w = 0.38
ax[1].bar(x - w/2, mod_a, w, color=LIGHT, label='Model score')
ax[1].bar(x + w/2, dis_a, w, color=TEAL, label='Disagreement')
for i, v in zip(x, mod_a):
    ax[1].text(i - w/2, v + 6, str(v), ha='center', fontsize=14, color=NAVY)
for i, v in zip(x, dis_a):
    ax[1].text(i + w/2, v + 6, str(v), ha='center', fontsize=14, fontweight='bold', color=TEAL)
ax[1].set_xticks(x); ax[1].set_xticklabels(bud); ax[1].set_ylim(0, 410)
ax[1].set_xlabel('Alert budget (alerts per 1,000 instances)')
ax[1].set_ylabel('ALL events caught'); ax[1].legend(frameon=False, loc='upper left')
titled(ax[1], 'B', 'The cost: fewer total events')

g = np.arange(2); w = 0.36
full = [0.682, 0.768]; phys = [0.648, 0.666]
ax[2].bar(g - w/2, full, w, color=NAVY, label='Full model')
ax[2].bar(g + w/2, phys, w, color=ORANGE, label='Physiology only')
for i, (a, b) in enumerate(zip(full, phys)):
    ax[2].text(i - w/2, a + .004, f'{a:.3f}', ha='center', fontsize=14, fontweight='bold', color=NAVY)
    ax[2].text(i + w/2, b + .004, f'{b:.3f}', ha='center', fontsize=14, fontweight='bold', color=ORANGE)
ax[2].set_xticks(g)
ax[2].set_xticklabels(['Low concern\n(nobody watching)', 'High concern\n(already watched)'])
ax[2].set_ylim(0.5, 0.81); ax[2].set_ylabel('AUROC within stratum')
ax[2].legend(frameon=False, loc='upper left')
titled(ax[2], 'C', 'There is real signal among the unwatched')
fig.tight_layout(); fig.savefig(OUT + 'new1.png', bbox_inches='tight'); plt.close(fig)

# ======================================================== Figure 4.3 ====
fig, ax = plt.subplots(1, 3, figsize=(2914/110, 695/110))
surv = [21827, 1512]; died = [1493, 568]
ax[0].bar([0, 1], surv, 0.55, color=LIGHT, label='Survived')
ax[0].bar([0, 1], died, 0.55, bottom=surv, color=RED, label='Died in hospital')
ax[0].text(0, surv[0] + died[0]/2, '1,493', ha='center', va='center', fontsize=16,
           fontweight='bold', color='white')
ax[0].text(0, surv[0]/2, '21,827', ha='center', va='center', fontsize=15, color=NAVY)
ax[0].text(1.33, surv[1] + died[1], '568 died', ha='right', fontsize=15, fontweight='bold', color=RED)
ax[0].text(1.33, surv[1] - 900, '1,512 survived', ha='right', fontsize=14, color=NAVY)
ax[0].text(-0.33, 24600, '1,493 of 2,061 deaths\nsit in this column', fontsize=13.5,
           style='italic', color=RED)
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(['No escalation', 'Escalation'])
ax[0].set_ylim(0, 27500); ax[0].set_ylabel('ICU stays contributing windows')
ax[0].legend(frameon=False, loc='upper right')
titled(ax[0], 'A', '72% of deaths are labelled "no event"')

grp = ['Elective\nadmission', 'Emergency\nadmission', 'CVICU', 'Neuro\nIntermediate']
whole = [76.2, 47.9, 86.6, 2.7]; after = [0.76, 0.76, 1.66, 0.10]
ax[1].bar(np.arange(4), whole, 0.55, color=LIGHT)
for i, v in enumerate(whole):
    ax[1].text(i, v + 1.5, f'{v}%', ha='center', fontsize=14, color=GREY)
ax[1].texts[-1].set_position((2.58, 3.2))
ax2 = ax[1].twinx(); ax2.spines['top'].set_visible(False)
ax2.plot(np.arange(4), after, '-o', color=TEAL, lw=3, ms=12)
for i, v in enumerate(after):
    ax2.text(i + (0.16 if i == 3 else 0), v + .09, f'{v}%', ha='center', fontsize=14,
             fontweight='bold', color=TEAL)
ax2.set_ylim(0, 2.5); ax2.set_ylabel('Event rate, modelled windows (%)', color=TEAL)
ax2.tick_params(axis='y', colors=TEAL)
ax[1].set_xticks(np.arange(4)); ax[1].set_xticklabels(grp)
ax[1].set_ylim(0, 100); ax[1].set_ylabel('Escalation rate, whole stay (%)')
ax[1].text(0, 94, 'elective 76.2% → 0.76%: identical to emergency', fontsize=13.5,
           style='italic', color=TEAL)
titled(ax[1], 'B', 'The landmark removes planned escalation')

units = ['CVICU', 'CCU', 'SICU', 'TSICU', 'MICU/\nSICU', 'MICU']
vals = [0.854, 0.813, 0.757, 0.747, 0.734, 0.721]
sds = [0.027, 0.014, 0.027, 0.019, 0.031, 0.031]
cols_ = [GREEN, GREEN, ORANGE, ORANGE, RED, RED]
y = np.arange(len(units))[::-1]
ax[2].barh(y, vals, 0.55, color=cols_, xerr=sds, error_kw=dict(ecolor='#555', lw=1.6, capsize=4))
for yy, v, sd in zip(y, vals, sds):
    ax[2].text(0.895, yy, f'{v:.3f}', va='center', ha='left', fontsize=14.5,
               fontweight='bold', color=NAVY)
ax[2].axvline(0.783, ls='--', color=NAVY, lw=2)
ax[2].set_ylim(-1.15, 5.75)
ax[2].text(0.792, -0.72, 'pooled 0.783', fontsize=13.5, color=NAVY, ha='left', va='center')
ax[2].set_yticks(y); ax[2].set_yticklabels(units)
ax[2].set_xlim(0.5, 0.96); ax[2].set_xlabel('AUROC within care unit')
ax[2].text(0.505, y.max() + 0.5, 'MICU is the classic deterioration population',
           fontsize=13.5, style='italic', color=GREY)
titled(ax[2], 'C', 'Worst where it is needed most')
fig.tight_layout(); fig.savefig(OUT + 'new2.png', bbox_inches='tight'); plt.close(fig)
print('wrote new1.png (Fig 4.2), new2.png (Fig 4.3), new3.png (Fig 4.1)')
