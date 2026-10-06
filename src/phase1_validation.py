"""False positives on the complete, unseen Phase 1 benign traffic (Table 9, Fig. 5).

The models are trained once on the Phase 2 sample (40,000 benign + 1,004 attacks).
Phase 1 is never used for training or tuning; it is read in chunks and only predicted.
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from common import (PHASE1, load_binary, make_prep, borderline_smote,
                    models_a, random_forest_a, stacking_a)

warnings.filterwarnings('ignore', category=UserWarning)
CHUNK = 500_000

df, features = load_binary()
X, y = df[features].to_numpy(), df['label'].astype(int).to_numpy()
del df

prep = make_prep()
X_sm, y_sm = borderline_smote().fit_resample(prep.fit_transform(X, y), y)
keep = prep.named_steps['select'].get_support()
print("selected features:", [f for f, k in zip(features, keep) if k])

base = list(models_a())
models = {'RandomForest': random_forest_a(), **models_a(), 'Stacking': stacking_a()}
train_s = {}
for name, m in models.items():
    t0 = time.time()
    m.fit(X_sm, y_sm)
    train_s[name] = time.time() - t0
train_s['Voting (soft)'] = train_s['Voting (hard)'] = sum(train_s[n] for n in base)

order = list(models) + ['Voting (soft)', 'Voting (hard)']
fp = dict.fromkeys(order, 0)
infer_s = dict.fromkeys(order, 0.0)
flagged_by = np.zeros(len(order) + 1, dtype=int)   # packets flagged by k models
n_seen, not_benign = 0, 0

for chunk in pd.read_csv(PHASE1, usecols=features + ['label'], chunksize=CHUNK,
                         low_memory=False):
    not_benign += int((chunk['label'] != 0).sum())   # Phase 1 is benign only: expect 0
    Xp = prep.transform(chunk[features].to_numpy())
    proba, pred = {}, {}
    for name, m in models.items():
        t0 = time.time()
        proba[name] = m.predict_proba(Xp)[:, 1]
        infer_s[name] += time.time() - t0
        pred[name] = proba[name] >= 0.5
    pred['Voting (soft)'] = np.mean([proba[n] for n in base], axis=0) >= 0.5
    pred['Voting (hard)'] = np.sum([pred[n] for n in base], axis=0) >= 3   # 2-2 -> normal
    for name in order:
        fp[name] += int(pred[name].sum())
    flagged_by += np.bincount(np.sum([pred[n] for n in order], axis=0),
                              minlength=len(order) + 1)
    n_seen += len(Xp)
    print(f"{n_seen:>12,} packets  stacking FP {fp['Stacking']}")

for v in ['Voting (soft)', 'Voting (hard)']:
    infer_s[v] = sum(infer_s[n] for n in base)


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


rows = []
for name in order:
    lo, hi = clopper_pearson(fp[name], n_seen)
    rows.append({'model': name, 'false_positives': fp[name],
                 'FPR_%': 100 * fp[name] / n_seen,
                 'CI95_low_%': 100 * lo, 'CI95_high_%': 100 * hi,
                 'FP_per_million': 1e6 * fp[name] / n_seen,
                 'train_s': train_s[name], 'packets_per_s': n_seen / infer_s[name]})
res = pd.DataFrame(rows).sort_values('false_positives')
pd.set_option('display.width', 200)
print(f"\nPhase 1: {n_seen:,} packets, {not_benign} not labelled benign")
print(res.round(5).to_string(index=False))
res.to_csv("phase1_false_positives.csv", index=False)
print("packets flagged by k of the 8 models:",
      {k: int(c) for k, c in enumerate(flagged_by) if k > 0})
