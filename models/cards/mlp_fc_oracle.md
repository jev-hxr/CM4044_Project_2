# Model card - MLP (ECFP4), loss correction (oracle T)

**Role:** main model (defence against label noise)  
**Project:** CM4044 MP2-30 - label noise and robustness on Tox21

## Architecture and training
Multi-task MLP on 2048-bit ECFP4: Linear-BatchNorm-ReLU-Dropout blocks [1024, 256], 12 sigmoid outputs, masked BCE (missing labels ignored). Adam lr=0.001, weight decay=1e-05, batch 128, dropout 0.3, <= 80 epochs with early stopping (patience 12) on mean validation ROC-AUC. ~2.4 M parameters.

## Noise handling
Forward loss correction with the *true* injected flip rates - an upper bound that isolates the effect of noise-channel estimation error. Not deployable in practice.

## Training data
`data/tox21_curated.csv` (sha256 prefix `b801bd1e6da05290`), 7,522 standardised parent structures, 12 Tox21 assays, deterministic Bemis-Murcko scaffold split 80/10/10 (see `data/curation_log.md`). Under a noise condition, train+valid labels are corrupted with `noisytox/noise.py` (seeded); the test set always keeps the original labels.

## Checkpoints
`models/checkpoints/mlp_fc_oracle_<condition>_s0.pt` (seed 0; `<condition>` = clean, sym_0.05 ... cc_0.20). Load with `torch.load(...)['state_dict']` into `noisytox.models.MLP` using the stored `hp`.

## Test performance (mean ± sd over seeds, clean test labels)
| condition | ROC-AUC | PR-AUC | labels removed |
|---|---|---|---|
| sym_0.10 | 0.666 ± nan | 0.275 ± nan | 0 |

## Intended use
Teaching / research: studying how label noise affects toxicity classifiers and comparing defences under matched conditions. Ranking compounds for *in vitro* Tox21-type assay activity only.

## Limitations
* Not a regulatory or safety tool; Tox21 assays are *in vitro* HTS readouts, not *in vivo* toxicity.
* Scaffold-split test set contains rarer chemotypes; per-assay test sets hold only 20-117 actives, so per-assay metrics are noisy (use the 12-assay mean and seed spread).
* Standardisation keeps the parent fragment only: counter-ion / mixture effects are ignored.
* Applicability domain: small organic molecules similar to the Tox21 10K library.
* Synthetic noise models (uniform / prior-preserving class-conditional) are simplifications of real assay error, which is structure-dependent.
