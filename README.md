# CM4044 Mini Project 2 · MP2-30 — How Noisy Is Toxicity?
### Label noise and ensemble robustness on Tox21

Toxicity labels come from real high-throughput assays with real error rates. This project measures
(1) how much **natural** noise Tox21 contains, (2) how fast an undefended MLP / GNN degrades under
**controlled synthetic** label noise, (3) which **defences** recover the loss, and (4) whether the best
noise detector red-flags **real suspect labels** (molecules whose related assays disagree) more often
than concordant ones.

**Baseline vs. main models.** The baseline is the *undefended* model (standard training). Each
defence is a main model, trained with identical splits, seeds, hyperparameters and metrics.

| Role | Model | Notes |
|---|---|---|
| Classical baseline | Random forest on ECFP4 | one forest per assay |
| Baseline (undefended) | multi-task MLP on ECFP4, GINE graph network | standard masked BCE |
| Main: label smoothing (`ls`) | MLP / GNN | ε = 0.1 |
| Main: loss correction (`fc`) | MLP / GNN | forward correction, noise channel estimated by confident learning |
| Main: loss correction, oracle (`fc_oracle`) | MLP / GNN | true flip rates; an upper bound that isolates estimation error |
| Main: confident learning (`cl`) | RF / MLP / GNN | cleanlab `prune_by_noise_rate` → flagged labels set to missing |
| Main: ensemble-disagreement filter (`ed`) | RF / MLP / GNN | majority vote of 3 diverse out-of-fold learners |

---

## Quick start (Google Colab, free T4)

1. Open `notebooks/MP2-30_Tox21_label_noise.ipynb` in Colab (*File → Open notebook → GitHub*, or upload it).
2. Set *Runtime → Change runtime type → T4 GPU*.
3. Run the cells from top to bottom. The repository is cloned into Google Drive, and each training run
   is saved there as soon as it finishes. After a disconnect, re-run the same cell and it skips the
   runs that are already done.

To split the grid across the three group members, give everyone the same command with a different
`--seeds 0`, `--seeds 1` or `--seeds 2`. Then copy all the `results/runs/*.json` and
`results/detection/*.json` files into one folder and run `python -m noisytox.aggregate`.

### Running locally (CPU works, GPU optional)
```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
python -m noisytox.curate                # data/tox21_curated.csv + data/curation_log.md
python -m noisytox.natural_noise         # Task 1
python -m noisytox.run_grid              # Tasks 2+3 (full grid; see --help for subsets)
python -m noisytox.aggregate             # tables + figures
python -m noisytox.validate_suspects     # Task 4
python -m noisytox.interaction           # stretch goal
python -m noisytox.model_cards           # models/cards/*.md
```

### Compute budget (measured)
| Step | 4-core CPU (dev machine) | Colab T4 (estimate) |
|---|---|---|
| Curation + features | 25 s | 30 s |
| Detector OOF probabilities, per noise condition | ~100 s | ~60–90 s |
| MLP run | ~10 s | ~5 s |
| RF run (100 trees × 12 assays) | ~25–40 s | ~60 s (2 vCPUs) |
| GNN run | ~4–6 min | ~1 min |
| Full grid: 7 conditions × 3 seeds × (3 RF + 6 MLP + 6 GNN runs) | — | ≈ 4–5 h in total, resumable |

---

## Repository layout
```
noisytox/
  config.py            every path, task name and hyperparameter
  curate.py            standardisation, deduplication, scaffold split, curation log
  data.py              dataset loader, ECFP4 + PyG graph featurisation (cached)
  noise.py             synthetic noise injection (symmetric / class-conditional)
  models.py            MLP and GINE architectures
  train.py             masked multi-task training, label smoothing, forward correction, RF
  detectors.py         out-of-fold probabilities, confident learning, ensemble-disagreement filter
  natural_noise.py     Task 1: imbalance, replicate and cross-assay disagreement
  run_grid.py          Tasks 2+3: noise x defence x model x seed grid (resumable)
  aggregate.py         summary tables, comparison table, figures
  validate_suspects.py Task 4: detector flags vs. real assay-disagreement suspects
  interaction.py       stretch: multi-task noise interaction
  model_cards.py       model-card generator
  versions.py          records package versions to results/versions.json
notebooks/MP2-30_Tox21_label_noise.ipynb   Colab driver notebook
data/raw/tox21.csv.gz                      raw MoleculeNet file (sha256-checked)
data/tox21_curated.csv                     curated dataset (deliverable 3)
data/curation_log.md                       every curation decision with record counts
data/duplicate_conflicts.csv               replicate label conflicts (natural noise)
results/                                   CSV / markdown outputs (comparison_table.md = slide table)
figures/                                   PNG figures for the slides
models/cards/                              model cards; checkpoints go to models/checkpoints/
docs/slides_outline.md                     12-slide outline
```

---

## Methods (and why)

**Curation** (`data/curation_log.md`). RDKit cleanup, largest organic fragment, neutralisation and
canonical SMILES. Records that do not parse, contain no carbon or have fewer than 3 heavy atoms are
dropped. Duplicate parent structures are merged: labels that agree are kept, conflicts with a strict
majority keep the majority label, and ties become missing. Every conflict is logged, because
disagreement between replicates *is* natural label noise. The result is 7,522 compounds, within the
≤10k budget, so no subsampling was needed. The split is a deterministic Bemis–Murcko scaffold split
(80/10/10), which tests generalisation to new chemotypes.

**Natural noise (Task 1).** Three views:
1. Class imbalance per assay (1:5 to 1:35).
2. Replicate disagreement: the same parent compound measured twice with different results.
3. Cross-assay disagreement between assays that probe the same receptor:
   - NR-AR (full-length receptor, MDA cell line) vs. NR-AR-LBD (ligand-binding domain);
   - NR-ER (full-length, BG1 cell line, luciferase) vs. NR-ER-LBD.

   A molecule active in one assay and inactive in its partner is a **suspect label**.

**Synthetic noise (Task 2)**, in `noisytox/noise.py`, injected into train + valid labels only.
The test set always keeps its original labels.
* `sym`: exactly *p* × (number of labelled molecules) labels per assay are flipped uniformly at random.
  With about 5 % actives, this mostly creates *false actives*.
* `cc`, class-conditional and prior-preserving: a fraction *p* of the actives become inactive (missed
  actives), and the same number of inactives become active (assay artefacts). The class balance looks
  unchanged, so the noise is invisible in the label statistics.
* The validation set used for early stopping is also noisy, because a real modeller never has clean
  labels to tune on.

**Defences (Task 3)** all share one detector pipeline:
* **Out-of-fold probabilities.** Logistic regression, Tanimoto kNN and an MLP are each trained with
  5-fold cross-validation, so every label is scored by models that never saw it.
* **Confident learning** (Northcutt et al. 2021) uses `cleanlab` on the averaged probabilities. It
  also estimates the per-assay noise transition matrix, which forward loss correction uses
  (Patrini et al. 2017).
* **Ensemble-disagreement filter.** Each learner votes a label wrong when it confidently predicts the
  other class, using class-specific thresholds. A label with ≥ 2 of 3 votes is removed.
* **How cleaning removes labels.** A flagged label is set to *missing* for that assay only, so the
  molecule still trains the other 11 heads.

**Task 4.** The detectors run on the natural labels of all molecules. The flag rate of suspect
labels is compared with that of concordant labels:
* within each label class (actives vs. actives, inactives vs. inactives), because suspects are
  enriched in actives;
* using a one-sided Fisher exact test, lift, precision/recall, and the AUC of the continuous
  detector score.

**Stretch goal.** 20 % noise is injected into the 7 NR assays only, and the effect on the 5 clean SR
assays is measured in a shared network under four regimes: clean, noisy, noisy + CL cleaning, and
NR labels dropped.

**Metrics.** Test ROC-AUC and PR-AUC (average precision) are computed per assay and averaged over the
12 assays. They are reported as mean ± sd over 3 seeds, plus the paired difference against the
undefended model with the same seed.

## Reproducibility
* Fixed seeds: the noise draw uses seed `10000+s`, model initialisation and batch order use seed `s`,
  and the CV folds are seeded. The split is deterministic. The raw file is checked against its SHA-256.
* `results/versions.json` records the exact package versions.
* Exact bit-for-bit GPU reproducibility is not guaranteed, because PyG scatter operations use atomic
  adds on CUDA. Differences are far below the seed-to-seed spread.

## References
1. Mayr, A.; Klambauer, G.; Unterthiner, T.; Hochreiter, S. DeepTox: Toxicity prediction using deep learning. *Front. Environ. Sci.* 2016, 3, 80.
2. Northcutt, C. G.; Jiang, L.; Chuang, I. L. Confident learning: Estimating uncertainty in dataset labels. *J. Artif. Intell. Res.* 2021, 70, 1373–1411.
3. Wu, Z.; Ramsundar, B.; Feinberg, E. N.; et al. MoleculeNet: a benchmark for molecular machine learning. *Chem. Sci.* 2018, 9, 513–530.
4. Patrini, G.; Rozza, A.; Menon, A. K.; Nock, R.; Qu, L. Making deep neural networks robust to label noise: a loss correction approach. *CVPR* 2017.
5. Brodley, C. E.; Friedl, M. A. Identifying mislabeled training data. *J. Artif. Intell. Res.* 1999, 11, 131–167.
6. Hu, W.; Liu, B.; Gomes, J.; et al. Strategies for pre-training graph neural networks. *ICLR* 2020 (GINE).
