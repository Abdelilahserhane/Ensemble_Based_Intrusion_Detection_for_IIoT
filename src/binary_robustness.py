"""Repeated 5 x 10-fold CV of the binary stage and paired tests (Table 10, Section IV-B).

    python binary_robustness.py A    # configuration A: 4 base learners + Random Forest
    python binary_robustness.py B    # configuration B: 5 base learners (MLP included)

Configuration A also evaluates the two refinements reported in Section IV-B:
a weighted soft voting and a decision threshold tuned for every model. Both are
chosen on a validation set made of 15% of each training fold, so in this
configuration the models are trained on the remaining 85%. The test fold is
never used for any choice.
"""
import sys
import time
import itertools
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (f1_score, confusion_matrix, precision_recall_curve,
                             average_precision_score)

from common import (SEED, load_binary, make_prep, borderline_smote, models_a,
                    random_forest_a, stacking_a, models_b, stacking_b, compare)

warnings.filterwarnings('ignore', category=ConvergenceWarning)

CONFIG = sys.argv[1].upper() if len(sys.argv) > 1 else 'A'
N_REPEATS, N_SPLITS = 5, 10

# weights tried for the weighted voting; equal weights win ties
WEIGHTS = [(1, 1, 1, 1)] + [w for w in itertools.product(range(4), repeat=4)
                            if sum(w) > 0 and w != (1, 1, 1, 1)]


def best_threshold(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1))])


def best_weights(y_val, P_val):
    best, best_score = None, -1.0
    for w in WEIGHTS:
        score = average_precision_score(y_val, P_val @ np.array(w) / sum(w))
        if score > best_score + 1e-9:
            best, best_score = w, score
    return np.array(best, dtype=float)


df, features = load_binary()
X, y = df[features].to_numpy(), df['label'].astype(int).to_numpy()

rows, weights_log = [], []


def record(rep, fold, name, y_te, pred, threshold='0.5'):
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    rows.append({'repeat': rep, 'fold': fold, 'model': name, 'threshold': threshold,
                 'f1': f1_score(y_te, pred), 'FP': fp, 'FN': fn})


t_start = time.time()
for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_fit, y_fit, y_te = X[tr], y[tr], y[te]
        if CONFIG == 'A':
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_fit, y_fit, test_size=0.15, stratify=y_fit,
                random_state=SEED + 100 * rep + fold)
        prep = make_prep()
        X_fit_p = prep.fit_transform(X_fit, y_fit)
        X_te_p = prep.transform(X[te])
        X_sm, y_sm = borderline_smote().fit_resample(X_fit_p, y_fit)
        if CONFIG == 'A':
            X_val_p = prep.transform(X_val)

        models = models_a() if CONFIG == 'A' else models_b()
        base = list(models)
        if CONFIG == 'A':
            models['RandomForest'] = random_forest_a()
        models['Stacking'] = stacking_a() if CONFIG == 'A' else stacking_b()

        pt, pv = {}, {}
        for name, m in models.items():
            m.fit(X_sm, y_sm)
            pt[name] = m.predict_proba(X_te_p)[:, 1]
            if CONFIG == 'A':
                pv[name] = m.predict_proba(X_val_p)[:, 1]
        pt['Voting (soft)'] = np.mean([pt[n] for n in base], axis=0)
        votes = np.sum([pt[n] >= 0.5 for n in base], axis=0)
        record(rep, fold, 'Voting (hard)', y_te, (votes > len(base) / 2).astype(int))

        if CONFIG == 'A':
            pv['Voting (soft)'] = np.mean([pv[n] for n in base], axis=0)
            w = best_weights(y_val, np.column_stack([pv[n] for n in base]))
            weights_log.append(w)
            pt['Voting (weighted)'] = np.column_stack([pt[n] for n in base]) @ w / w.sum()
            pv['Voting (weighted)'] = np.column_stack([pv[n] for n in base]) @ w / w.sum()
            st = StackingClassifier(
                [(k.lower(), m) for k, m in models_a().items()],
                final_estimator=LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                                   random_state=SEED),
                cv=3, passthrough=True, n_jobs=-1).fit(X_sm, y_sm)
            pt['Stacking (passthrough)'] = st.predict_proba(X_te_p)[:, 1]
            pv['Stacking (passthrough)'] = st.predict_proba(X_val_p)[:, 1]

        for name, p in pt.items():
            record(rep, fold, name, y_te, (p >= 0.5).astype(int))
            if CONFIG == 'A':
                thr = best_threshold(y_val, pv[name])
                record(rep, fold, name, y_te, (p >= thr).astype(int), 'tuned')
        print(f"repeat {rep + 1} fold {fold:2d}  ({(time.time() - t_start) / 60:.1f} min)")

res = pd.DataFrame(rows).sort_values(['threshold', 'model', 'repeat', 'fold'])
res.to_csv(f"binary_robustness_{CONFIG}_folds.csv", index=False)
pd.set_option('display.width', 200)

summary = (res.groupby(['threshold', 'model'])
              .agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'), worst_fold=('f1', 'min'),
                   FN=('FN', 'sum'), FP=('FP', 'sum'))
              .sort_values(['threshold', 'f1_mean'], ascending=[True, False]))
print(summary.round(4).to_string())
if weights_log:
    print("mean weights (XGBoost, LightGBM, CatBoost, ExtraTrees):",
          np.round(np.mean(weights_log, axis=0), 2))

# paired tests at the default threshold, one Holm family per configuration
default = res[res.threshold == '0.5']
scores = {m: g['f1'].to_numpy() for m, g in default.groupby('model')}
ensembles = [m for m in scores if m.startswith(('Voting', 'Stacking'))]
individuals = [m for m in scores if m not in ensembles]
tests = compare(scores, ensembles, individuals, test_train_ratio=1 / (N_SPLITS - 1))
print(tests.round(4).to_string(index=False))
tests.to_csv(f"binary_robustness_{CONFIG}_tests.csv", index=False)
