"""Binary stage on the held-out real test set, plus 10-fold CV (Table 8).

Split first (70/30, stratified), then every data-dependent step lives inside an
imblearn pipeline: feature selection, scaling and BorderlineSMOTE are fitted on
the training part only. The test set is never resampled.
"""
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from imblearn.pipeline import Pipeline

from common import (SEED, load_binary, make_prep, borderline_smote,
                    models_a, random_forest_a, mlp_a, stacking_a)

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def pipeline(clf):
    prep = make_prep()
    return Pipeline(prep.steps + [('smote', borderline_smote()), ('clf', clf)])


def voting(mode):
    return VotingClassifier(list(models_a().items()), voting=mode, n_jobs=-1)


MODELS = {
    'RandomForest': random_forest_a,
    'XGBoost': lambda: models_a()['XGBoost'],
    'LightGBM': lambda: models_a()['LightGBM'],
    'CatBoost': lambda: models_a()['CatBoost'],
    'ExtraTrees': lambda: models_a()['ExtraTrees'],
    'MLP': mlp_a,
    'Voting (soft)': lambda: voting('soft'),
    'Voting (hard)': lambda: voting('hard'),
    'Stacking': stacking_a,
}

df, features = load_binary()
X, y = df[features].to_numpy(), df['label'].astype(int).to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=SEED)
print(f"train {len(y_tr)} ({y_tr.sum()} attacks), test {len(y_te)} ({y_te.sum()} attacks)")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
rows = []
for name, make in MODELS.items():
    pipe = pipeline(make())
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_s = time.time() - t0
    t0 = time.time()
    pred = pipe.predict(X_te)
    pred_s = time.time() - t0

    has_proba = name != 'Voting (hard)'          # hard voting gives no probabilities
    proba = pipe.predict_proba(X_te)[:, 1] if has_proba else None
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    cv_f1 = cross_validate(pipeline(make()), X, y, cv=cv, scoring='f1')['test_score']
    rows.append({
        'model': name,
        'accuracy': accuracy_score(y_te, pred),
        'precision': precision_score(y_te, pred),
        'recall': recall_score(y_te, pred),
        'f1': f1_score(y_te, pred),
        'roc_auc': roc_auc_score(y_te, proba) if has_proba else np.nan,
        'pr_auc': average_precision_score(y_te, proba) if has_proba else np.nan,
        'FP': fp, 'FN': fn,
        'cv_f1_mean': cv_f1.mean(), 'cv_f1_std': cv_f1.std(),
        'train_s': train_s, 'predict_s': pred_s,
    })
    print(f"{name:<14s} F1 {rows[-1]['f1']:.4f}  FP/FN {fp}/{fn}  "
          f"CV F1 {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(res.round(4).to_string(index=False))
res.to_csv("binary_test_set.csv", index=False)
