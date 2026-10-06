# Ensemble-Based Intrusion Detection for IIoT

Code for the paper *Ensemble-Based Intrusion Detection for IIoT: A Two-Stage
Classification Framework with Feature Selection and Data Balancing*
(A. Serhane, K. Ibrahimi, E.-M. Hamzaoui, submitted to IEEE Access).

The repository contains only what is needed to reproduce the final results
reported in the paper. The exploratory comparisons (dimensionality reduction
methods, SMOTE variants) are not included.

## Data

We use the network traffic of the CIC-APT IIoT 2024 dataset
(Ghiasvand et al., 2024, [arXiv:2407.11278](https://arxiv.org/abs/2407.11278)),
distributed by the Canadian Institute for Cybersecurity. Two files are needed:

- `phase2_NetworkData.csv`: traffic recorded during the APT campaign (training and testing)
- `phase1_NetworkData.csv`: benign traffic only, used for the final false-positive check

The experiments were run on Kaggle, where the files sit in `/kaggle/input/cicapt-iiot/`.
Elsewhere, point the scripts to the folder that holds them:

```bash
export CICAPT_DIR=/path/to/cicapt-iiot
```

## Protocol in short

- Identifier fields (timestamp, IP addresses, ports, protocol name, MAC) are dropped,
  leaving 60 numerical features.
- Binary stage: 40,000 benign packets sampled at random from Phase 2 plus all
  1,004 attacks. Multiclass stage: the 1,004 attacks, labelled by technique
  (`subLabelCat`, 25 classes).
- The data are split first. Feature selection (top 20 by Random Forest importance),
  RobustScaler and oversampling (BorderlineSMOTE for the binary stage, SMOTE for the
  multiclass stage) are then fitted on the training part only, inside every fold.
  Test data are never resampled.
- Hyperparameters: configuration A (binary) and configuration B (multiclass), as in
  Table 7. They are defined once in `src/common.py`. The random seed is 42 everywhere.

## Scripts and results

| Script | Paper |
|---|---|
| `src/binary_test_set.py` | Table 8 (tree-based models, MLP, voting and stacking on the real test set, 10-fold CV) |
| `src/binary_cnn1d.py` | Table 8, CNN 1D row |
| `src/feature_relevance.py` | Table 6 |
| `src/phase1_validation.py` | Table 9 and Fig. 5 (false positives on the 12.06 M Phase 1 packets) |
| `src/binary_robustness.py A` and `B` | Table 10 and the statistical tests of Section IV-B |
| `src/unseen_attacks.py` | Table 11 (leave-one-technique-out and leave-one-tactic-out) |
| `src/multiclass.py` | Tables 12 and 13, Fig. 6 |

Run them from the `src` folder, for example:

```bash
cd src
python binary_test_set.py
python binary_robustness.py A
```

Each script prints its results and writes them to CSV files in the current folder.
Indicative run times on a 4-core CPU: a few minutes for the binary test set and the
feature relevance, about 10 minutes for the multiclass stage, 25 minutes for the
unseen attacks, and several hours for the repeated cross-validation of
`binary_robustness.py`.

## Installation

```bash
pip install -r requirements.txt
```

TensorFlow is only needed for `binary_cnn1d.py`.

## Note on reproducibility

All models were trained on CPU with fixed seeds. XGBoost and LightGBM build their
trees with several threads, and a different machine or library version can change
a few predictions. When we ran the code a second time, almost every figure was
identical, but a couple of single-split results moved by a few packets (this is
discussed in Section III-C of the paper). The cross-validated results and the
statistical tests are much less sensitive to this.
