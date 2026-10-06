"""Relevance of the 20 features selected for the binary stage (Table 6).

Random Forest importance, Mutual Information and ANOVA F-score, all computed on
the training part of the binary split. Which of these features are also kept in
the multiclass stage is printed at the end of multiclass.py.
"""
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

from common import SEED, load_binary

df, features = load_binary()
X, y = df[features].to_numpy(), df['label'].astype(int).to_numpy()
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

X_tr = SimpleImputer(strategy='median').fit_transform(X_tr)
rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
top = np.argsort(rf.feature_importances_)[::-1][:20]

table = pd.DataFrame({
    'feature': np.array(features)[top],
    'rf_importance': rf.feature_importances_[top],
    'mutual_information': mutual_info_classif(X_tr[:, top], y_tr, random_state=SEED),
    'anova_f': f_classif(X_tr[:, top], y_tr)[0],
})
print(table.round(4).to_string(index=False))
table.to_csv("feature_relevance.csv", index=False)
