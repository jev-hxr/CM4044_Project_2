# Model card - RF (ECFP4), undefended (baseline)

**Role:** baseline  
**Project:** CM4044 MP2-30 - label noise and robustness on Tox21

## Architecture and training
Random forest, one per assay, on 2048-bit ECFP4 (Morgan radius 2) fingerprints. Hyperparameters: {'n_estimators': 100, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_jobs': -1}.

## Noise handling
Standard training on the (possibly noisy) labels - the **baseline**.

## Training data
`data/tox21_curated.csv` (sha256 prefix `b801bd1e6da05290`), 7,522 standardised parent structures, 12 Tox21 assays, deterministic Bemis-Murcko scaffold split 80/10/10 (see `data/curation_log.md`). Under a noise condition, train+valid labels are corrupted with `noisytox/noise.py` (seeded); the test set always keeps the original labels.

## Checkpoints
`models/checkpoints/rf_none_<condition>_s0.joblib` (seed 0; `<condition>` = clean, sym_0.05 ... cc_0.20). Load with `joblib.load` -> list of 12 scikit-learn forests (one per assay).

## Test performance (mean ± sd over seeds, clean test labels)
| condition | ROC-AUC | PR-AUC | labels removed |
|---|---|---|---|
| clean | 0.688 ± nan | 0.335 ± nan | 0 |
| sym_0.10 | 0.649 ± nan | 0.279 ± nan | 0 |

## Intended use
Teaching / research: studying how label noise affects toxicity classifiers and comparing defences under matched conditions. Ranking compounds for *in vitro* Tox21-type assay activity only.

## Limitations
* Not a regulatory or safety tool; Tox21 assays are *in vitro* HTS readouts, not *in vivo* toxicity.
* Scaffold-split test set contains rarer chemotypes; per-assay test sets hold only 20-117 actives, so per-assay metrics are noisy (use the 12-assay mean and seed spread).
* Standardisation keeps the parent fragment only: counter-ion / mixture effects are ignored.
* Applicability domain: small organic molecules similar to the Tox21 10K library.
* Synthetic noise models (uniform / prior-preserving class-conditional) are simplifications of real assay error, which is structure-dependent.
