"""CNN 1D baseline for the binary stage (Table 8), same split and preprocessing."""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

from common import SEED, load_binary, make_prep, borderline_smote

EPOCHS, BATCH_SIZE = 20, 64
tf.keras.utils.set_random_seed(SEED)


def prepare(X_tr, y_tr, X_te):
    prep = make_prep()
    X_tr = prep.fit_transform(X_tr, y_tr)
    X_te = prep.transform(X_te)
    X_tr, y_tr = borderline_smote().fit_resample(X_tr, y_tr)
    return X_tr[..., None], y_tr, X_te[..., None]      # (n, 20) -> (n, 20, 1)


def cnn(n_features):
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(32, kernel_size=2, activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit_predict(X_tr, y_tr, X_te):
    model = cnn(X_tr.shape[1])
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model.predict(X_te, verbose=0).ravel(), model


df, features = load_binary()
X, y = df[features].to_numpy(), df['label'].astype(int).to_numpy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                          random_state=SEED)

t0 = time.time()
A, b, T = prepare(X_tr, y_tr, X_te)
proba, model = fit_predict(A, b, T)
pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"trainable parameters: {model.count_params()}  ({time.time() - t0:.1f} s)")
print(f"accuracy {accuracy_score(y_te, pred):.4f}  precision {precision_score(y_te, pred):.4f}  "
      f"recall {recall_score(y_te, pred):.4f}  F1 {f1_score(y_te, pred):.4f}")
print(f"ROC-AUC {roc_auc_score(y_te, proba):.4f}  PR-AUC {average_precision_score(y_te, proba):.4f}"
      f"  FP/FN {fp}/{fn}")

# 10-fold CV, preprocessing and oversampling redone inside each fold
f1s = []
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for fold, (tr, te) in enumerate(cv.split(X, y), 1):
    tf.keras.utils.set_random_seed(SEED + fold)
    A, b, T = prepare(X[tr], y[tr], X[te])
    p, _ = fit_predict(A, b, T)
    f1s.append(f1_score(y[te], (p >= 0.5).astype(int)))
print(f"CV F1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
