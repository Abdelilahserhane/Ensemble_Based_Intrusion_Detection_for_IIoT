"""Multiclass stage: identify the attack technique (Tables 12 and 13, Fig. 6).

Only the 1,004 malicious packets are used (25 techniques, subLabelCat).
Repeated 5 x 5 stratified CV: the rarest techniques have 14 packets, so 5 folds
keep about 3 of them in each test fold. In every fold, feature selection,
scaling and SMOTE are fitted on the training part only.
"""
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix)

from common import SEED, load_attacks, make_prep, smote, models_b, stacking_b, compare

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

N_REPEATS, N_SPLITS = 5, 5

df, features = load_attacks()
le = LabelEncoder()                         # text label -> integer, no ordering implied
y = le.fit_transform(df['subLabelCat'])
X = df[features].to_numpy()
K = len(le.classes_)
labels = np.arange(K)
print(f"{len(y)} attacks, {K} techniques")


def hard_vote(preds):
    """Majority vote; a tie goes to the lowest class index."""
    P = np.column_stack(preds)
    return np.array([np.bincount(r, minlength=K).argmax() for r in P])


folds, oof, selected = [], [], []
t_start = time.time()
for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        prep = make_prep()
        X_tr = prep.fit_transform(X[tr], y[tr])
        X_te = prep.transform(X[te])
        selected.extend(np.array(features)[prep.named_steps['select'].get_support()])
        X_rs, y_rs = smote().fit_resample(X_tr, y[tr])

        proba, fit_s, inf_s = {}, {}, {}
        models = models_b(K)
        models['Stacking'] = stacking_b(K)
        for name, m in models.items():
            t0 = time.time()
            m.fit(X_rs, y_rs)
            fit_s[name] = time.time() - t0
            t0 = time.time()
            proba[name] = m.predict_proba(X_te)
            inf_s[name] = time.time() - t0

        base = [n for n in models if n != 'Stacking']
        pred = {n: p.argmax(1) for n, p in proba.items()}
        pred['Voting (soft)'] = np.mean([proba[n] for n in base], axis=0).argmax(1)
        pred['Voting (hard)'] = hard_vote([pred[n] for n in base])
        for v in ['Voting (soft)', 'Voting (hard)']:     # cost of the five base learners
            fit_s[v], inf_s[v] = sum(fit_s[n] for n in base), sum(inf_s[n] for n in base)

        for name, p in pred.items():
            folds.append({'repeat': rep, 'fold': fold, 'model': name,
                          'f1_macro': f1_score(y[te], p, labels=labels, average='macro',
                                               zero_division=0),
                          'balanced_accuracy': balanced_accuracy_score(y[te], p),
                          'accuracy': accuracy_score(y[te], p),
                          'train_s': fit_s[name], 'inference_s': inf_s[name]})
            oof.append(pd.DataFrame({'model': name, 'repeat': rep, 'idx': te,
                                     'y_true': y[te], 'y_pred': p}))
        print(f"repeat {rep + 1} fold {fold}  ({(time.time() - t_start) / 60:.1f} min)")

res = pd.DataFrame(folds).sort_values(['model', 'repeat', 'fold'])
oof = pd.concat(oof, ignore_index=True)
res.to_csv("multiclass_folds.csv", index=False)
pd.set_option('display.width', 200)

# Table 12
summary = (res.groupby('model')
              .agg(f1_macro=('f1_macro', 'mean'), f1_std=('f1_macro', 'std'),
                   worst_fold=('f1_macro', 'min'),
                   balanced_accuracy=('balanced_accuracy', 'mean'),
                   accuracy=('accuracy', 'mean'), train_s=('train_s', 'mean'),
                   inference_s=('inference_s', 'mean'))
              .sort_values('f1_macro', ascending=False))
print(summary.round(4).to_string())

scores = {m: g['f1_macro'].to_numpy() for m, g in res.groupby('model')}
ensembles = ['Voting (soft)', 'Voting (hard)', 'Stacking']
tests = compare(scores, ensembles, [m for m in scores if m not in ensembles],
                test_train_ratio=1 / (N_SPLITS - 1))
print(tests.round(4).to_string(index=False))
tests.to_csv("multiclass_tests.csv", index=False)

# Table 13: per-technique F1 on the out-of-fold predictions, averaged over repeats
per_tech = {}
for model, g in oof.groupby('model'):
    per_tech[model] = np.mean([f1_score(r.y_true, r.y_pred, labels=labels, average=None,
                                        zero_division=0)
                               for _, r in g.groupby('repeat')], axis=0)
per_tech = pd.DataFrame(per_tech, index=le.classes_)
per_tech.insert(0, 'packets', np.bincount(y, minlength=K))
per_tech = per_tech.sort_values('packets', ascending=False)
print(per_tech.round(3).to_string())
per_tech.to_csv("multiclass_per_technique_f1.csv")

# Fig. 6: confusion matrix of stacking, summed over the five repeats
st = oof[oof.model == 'Stacking']
cm = confusion_matrix(st.y_true, st.y_pred, labels=labels)
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(
    "multiclass_confusion_stacking.csv")

# stability of the selected features across the 25 folds
freq = pd.Series(selected).value_counts() / (N_REPEATS * N_SPLITS)
print("share of folds in which each feature is selected:")
print(freq.round(2).to_string())
