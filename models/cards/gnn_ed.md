# Model card - GIN (graph), ensemble-disagreement filter

**Role:** main model (defence against label noise)  
**Project:** CM4044 MP2-30 - label noise and robustness on Tox21

## Architecture and training
Multi-task GINE graph network (PyTorch Geometric): OGB-style categorical atom/bond embeddings, 4 residual GINEConv layers of width 128 with BatchNorm, mean+max pooling, 2-layer head, 12 sigmoid outputs, masked BCE. Adam lr=0.001, weight decay=1e-05, batch 128, dropout 0.2, <= 80 epochs, early stopping (patience 12). ~0.34 M parameters.

## Noise handling
Ensemble-disagreement (majority) filter: each of ['lr', 'knn', 'mlp'] votes a label wrong when its out-of-fold probability confidently favours the other class (class-specific thresholds); >= 2 votes -> label set to missing, then standard training.

## Training data
`data/tox21_curated.csv` (sha256 prefix `b801bd1e6da05290`), 7,522 standardised parent structures, 12 Tox21 assays, deterministic Bemis-Murcko scaffold split 80/10/10 (see `data/curation_log.md`). Under a noise condition, train+valid labels are corrupted with `noisytox/noise.py` (seeded); the test set always keeps the original labels.

## Checkpoints
`models/checkpoints/gnn_ed_<condition>_s0.pt` (seed 0; `<condition>` = clean, sym_0.05 ... cc_0.20). Load with `torch.load(...)['state_dict']` into `noisytox.models.GIN` using the stored `hp`.

## Intended use
Teaching / research: studying how label noise affects toxicity classifiers and comparing defences under matched conditions. Ranking compounds for *in vitro* Tox21-type assay activity only.

## Limitations
* Not a regulatory or safety tool; Tox21 assays are *in vitro* HTS readouts, not *in vivo* toxicity.
* Scaffold-split test set contains rarer chemotypes; per-assay test sets hold only 20-117 actives, so per-assay metrics are noisy (use the 12-assay mean and seed spread).
* Standardisation keeps the parent fragment only: counter-ion / mixture effects are ignored.
* Applicability domain: small organic molecules similar to the Tox21 10K library.
* Synthetic noise models (uniform / prior-preserving class-conditional) are simplifications of real assay error, which is structure-dependent.
