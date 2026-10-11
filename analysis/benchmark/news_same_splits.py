# -*- coding: utf-8 -*-
"""NEWS2 on the same five splits as the decomposition, so the increment reported
in Section 4.3.5 is a within-run quantity rather than a comparison across runs."""
import pandas as pd, numpy as np, json, importlib.util, sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

spec = importlib.util.spec_from_file_location('n', '/home/claude/news/news.py')
src = open('/home/claude/news/news.py').read()
ns = {}
exec(src.split('def news2')[0].replace('import ', 'import '), ns)   # imports only
exec('def news2' + src.split('def news2')[1].split('\n\n\n')[0], ns)
news2 = ns['news2']

B = '/mnt/user-data/uploads/outputs/cleaned_data/'
D = json.load(open('/home/claude/rerun/decomp.json'))
OUT = {}
for H in ('4h', '8h'):
    df = pd.read_csv(B + f'engineered_features_{H}_FULLSCALE.csv', low_memory=False)
    df['NEWS'] = news2(df)
    y = df['label'].values; g = df['subject_id'].values
    a, ap = [], []
    for s in range(5):
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=100 + s).split(df, y, g))
        a.append(roc_auc_score(y[te], df.iloc[te]['NEWS']))
        ap.append(average_precision_score(y[te], df.iloc[te]['NEWS']))
    full = D[H]['specified']['full']['auroc_mean']
    OUT[H] = dict(news_auroc=float(np.mean(a)), news_sd=float(np.std(a)),
                  news_auprc=float(np.mean(ap)),
                  full_auroc=full, increment=float(full - np.mean(a)),
                  increment_per_split=[round(full - x, 4) for x in a],
                  median_score=float(df.NEWS.median()),
                  pct_ge5=float(100 * (df.NEWS >= 5).mean()),
                  pct_ge7=float(100 * (df.NEWS >= 7).mean()))
    print(H, json.dumps(OUT[H], indent=2), flush=True)
json.dump(OUT, open('/home/claude/rerun/news.json', 'w'), indent=2)
print('wrote news.json')
