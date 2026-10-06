"""Shared pieces: data loading, leakage-free preprocessing, model settings, stats."""
import os
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

SEED = 42
DATA_DIR = os.environ.get("CICAPT_DIR", "/kaggle/input/cicapt-iiot")
PHASE1 = os.path.join(DATA_DIR, "phase1_NetworkData.csv")
PHASE2 = os.path.join(DATA_DIR, "phase2_NetworkData.csv")

# Identifiers are removed so that models only see traffic statistics.
ID_COLUMNS = ['ts', 'Source IP', 'Destination IP', 'Source Port',
              'Destination Port', 'Protocol_name', 'MAC']
LABEL_COLUMNS = ['label', 'subLabel', 'subLabelCat']


# ---------------------------------------------------------------- data
def load_binary():
    """40,000 random benign packets + all 1,004 attacks of Phase 2 (60 features)."""
    df = pd.read_csv(PHASE2, low_memory=False)
    df = df[df['label'].isin([0, 1])]
    normal = df[df['label'] == 0].sample(n=40000, random_state=SEED)
    attack = df[df['label'] == 1]
    df = pd.concat([normal, attack]).reset_index(drop=True)
    features = [c for c in df.columns if c not in ID_COLUMNS + LABEL_COLUMNS]
    return df, features


def load_attacks():
    """The 1,004 malicious packets of Phase 2, labelled by technique (subLabelCat)."""
    df = pd.read_csv(PHASE2, low_memory=False)
    df = df[df['label'] == 1]
    df = df[df['subLabelCat'].notna()].reset_index(drop=True)
    features = [c for c in df.columns if c not in ID_COLUMNS + LABEL_COLUMNS]
    return df, features


# ------------------------------------------------------- preprocessing
# Everything that learns from data is fitted on the training part only.
def make_prep():
    """Random Forest top-20 feature selection, then RobustScaler."""
    return SkPipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('select', SelectFromModel(RandomForestClassifier(random_state=SEED, n_jobs=-1),
                                   max_features=20, threshold=-np.inf)),
        ('scale', RobustScaler()),
    ])


def borderline_smote():          # binary stage
    return BorderlineSMOTE(k_neighbors=7, m_neighbors=5, kind='borderline-1',
                           random_state=SEED)


def smote():                     # multiclass stage
    return SMOTE(k_neighbors=5, random_state=SEED)


# --------------------------------------------------------------- models
# Configuration A: tuned for the binary stage (Table 7).
def models_a():
    return {
        'XGBoost': XGBClassifier(
            eval_metric='logloss', n_estimators=200, max_depth=5, learning_rate=0.2,
            subsample=1.0, gamma=0, colsample_bytree=1.0, random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMClassifier(
            n_estimators=300, max_depth=15, learning_rate=0.2, subsample=1.0,
            num_leaves=31, colsample_bytree=0.6, random_state=SEED, verbosity=-1, n_jobs=-1),
        'CatBoost': CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.2, l2_leaf_reg=1, border_count=64,
            loss_function='Logloss', random_seed=SEED, verbose=0, thread_count=-1),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, max_depth=None, min_samples_split=10, min_samples_leaf=1,
            max_features='log2', random_state=SEED, n_jobs=-1),
    }


def random_forest_a():
    return RandomForestClassifier(n_estimators=300, max_depth=None, max_features='sqrt',
                                  min_samples_leaf=1, min_samples_split=5,
                                  random_state=SEED, n_jobs=-1)


def mlp_a():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         max_iter=300, random_state=SEED)


def stacking_a():
    return StackingClassifier(
        estimators=[(k.lower(), m) for k, m in models_a().items()],
        final_estimator=LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                           random_state=SEED),
        cv=3, n_jobs=-1)


# Configuration B: tuned for the multiclass stage, the MLP is a fifth base learner.
def models_b(n_classes=2):
    multi = n_classes > 2
    xgb_obj = dict(objective='multi:softprob', num_class=n_classes,
                   eval_metric='mlogloss') if multi else dict(eval_metric='logloss')
    lgbm_obj = dict(objective='multiclass') if multi else {}
    cat_obj = dict(loss_function='MultiClass') if multi else {}
    return {
        'XGBoost': XGBClassifier(
            **xgb_obj, n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMClassifier(
            **lgbm_obj, n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, num_leaves=31, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1, n_jobs=-1),
        'CatBoost': CatBoostClassifier(
            **cat_obj, iterations=200, depth=4, learning_rate=0.1, l2_leaf_reg=1,
            border_count=32, random_seed=SEED, verbose=0, thread_count=-1),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1,
            max_features='sqrt', random_state=SEED, n_jobs=-1),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
            alpha=0.0001, max_iter=300, random_state=SEED),
    }


def stacking_b(n_classes=2):
    return StackingClassifier(
        estimators=[(k.lower(), m) for k, m in models_b(n_classes).items()],
        final_estimator=LogisticRegression(max_iter=200, random_state=SEED),
        cv=3, n_jobs=-1)


# ---------------------------------------------------------------- stats
def nadeau_bengio(d, test_train_ratio):
    """Corrected resampled t-test (Nadeau and Bengio, 2003), two-sided p-value."""
    n, var = len(d), np.var(d, ddof=1)
    if var == 0:
        return 1.0
    t = np.mean(d) / np.sqrt((1 / n + test_train_ratio) * var)
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def wilcoxon(a, b):
    try:
        return stats.wilcoxon(a, b, zero_method='zsplit').pvalue
    except ValueError:           # all differences are zero
        return 1.0


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare(scores, ensembles, individuals, test_train_ratio):
    """Paired tests of each ensemble against each individual learner.

    scores: dict model -> per-fold scores, in the same fold order for every model.
    """
    rows = []
    for ens in ensembles:
        for ind in individuals:
            a, b = np.asarray(scores[ens]), np.asarray(scores[ind])
            d = a - b
            rows.append({'ensemble': ens, 'individual': ind, 'mean_diff': d.mean(),
                         'ensemble_wins': int((d > 0).sum()), 'ties': int((d == 0).sum()),
                         'p_wilcoxon': wilcoxon(a, b),
                         'p_nadeau_bengio': nadeau_bengio(d, test_train_ratio)})
    out = pd.DataFrame(rows)
    out['p_wilcoxon_holm'] = holm(out['p_wilcoxon'])
    out['p_nadeau_bengio_holm'] = holm(out['p_nadeau_bengio'])
    return out
