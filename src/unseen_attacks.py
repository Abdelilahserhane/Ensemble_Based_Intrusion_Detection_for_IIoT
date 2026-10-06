"""Detection of attacks never seen in training (Table 11, Section IV-C).

For each technique (25) and each tactic (8), the group is removed from training:
    train = 70% of the benign packets + all other attacks
    test  = the remaining 30% of benign packets + every packet of the held-out group
Preprocessing and BorderlineSMOTE are fitted on the training part only.
"""
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from common import (SEED, load_binary, make_prep, borderline_smote, models_a,
                    random_forest_a, stacking_a, wilcoxon, holm)

warnings.filterwarnings('ignore', category=UserWarning)

df, features = load_binary()
normal = df[df['label'] == 0]
attack = df[df['label'] == 1].reset_index(drop=True)
Xn, Xa = normal[features].to_numpy(), attack[features].to_numpy()
n_tr, n_te = train_test_split(np.arange(len(Xn)), test_size=0.3, random_state=SEED)
base = list(models_a())

rows = []
t_start = time.time()
for level, column in [('technique', 'subLabelCat'), ('tactic', 'subLabel')]:
    labels = attack[column].astype(str).to_numpy()
    for group in sorted(np.unique(labels)):
        held = labels == group
        X_tr = np.vstack([Xn[n_tr], Xa[~held]])
        y_tr = np.r_[np.zeros(len(n_tr), int), np.ones((~held).sum(), int)]

        prep = make_prep()
        X_sm, y_sm = borderline_smote().fit_resample(prep.fit_transform(X_tr, y_tr), y_tr)
        X_att, X_ben = prep.transform(Xa[held]), prep.transform(Xn[n_te])

        models = {**models_a(), 'RandomForest': random_forest_a(), 'Stacking': stacking_a()}
        qa, qb = {}, {}                           # attack scores on held-out attacks / benign
        for name, m in models.items():
            m.fit(X_sm, y_sm)
            qa[name], qb[name] = m.predict_proba(X_att)[:, 1], m.predict_proba(X_ben)[:, 1]
        qa['Voting (soft)'] = np.mean([qa[n] for n in base], axis=0)
        qb['Voting (soft)'] = np.mean([qb[n] for n in base], axis=0)
        pa = {n: q >= 0.5 for n, q in qa.items()}
        pb = {n: q >= 0.5 for n, q in qb.items()}
        pa['Voting (hard)'] = np.sum([pa[n] for n in base], axis=0) >= 3   # 2-2 -> normal
        pb['Voting (hard)'] = np.sum([pb[n] for n in base], axis=0) >= 3

        for name in list(models) + ['Voting (soft)', 'Voting (hard)']:
            rows.append({'level': level, 'group': group, 'packets': int(held.sum()),
                         'model': name, 'detected': int(pa[name].sum()),
                         'detection_rate': float(pa[name].mean()),
                         'FPR_benign': float(pb[name].mean())})
        print(f"[{level}] {group:<40s} {held.sum():3d} packets  "
              f"({(time.time() - t_start) / 60:.1f} min)")

res = pd.DataFrame(rows)
res.to_csv("unseen_attacks_groups.csv", index=False)
pd.set_option('display.width', 200)

summary = (res.groupby(['level', 'model'])
              .apply(lambda s: pd.Series({
                  'mean_detection': s.detection_rate.mean(),          # each group counts once
                  'packet_detection': s.detected.sum() / s.packets.sum(),
                  'groups_fully_detected': int((s.detection_rate == 1).sum()),
                  'mean_FPR': s.FPR_benign.mean()}))
              .reset_index()
              .sort_values(['level', 'mean_detection'], ascending=[True, False]))
print(summary.round(4).to_string(index=False))
summary.to_csv("unseen_attacks_summary.csv", index=False)

# ensembles vs individual learners over the 25 techniques
tech = res[res.level == 'technique'].pivot_table(index='group', columns='model',
                                                 values='detection_rate')
tests = pd.DataFrame([{'ensemble': e, 'individual': i,
                       'mean_diff': (tech[e] - tech[i]).mean(),
                       'p_wilcoxon': wilcoxon(tech[e], tech[i])}
                      for e in ['Voting (soft)', 'Voting (hard)', 'Stacking']
                      for i in ['RandomForest'] + base])
tests['p_wilcoxon_holm'] = holm(tests['p_wilcoxon'])
print(tests.round(4).to_string(index=False))
