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

## Everything is in one notebook

**`notebooks/MP2-30_Tox21_label_noise.ipynb`** contains all of the project's code (~1,000 lines),
organised by task with an explanation above each section:

| Section | Content |
|---|---|
| 0 | Setup (package install, resume from zip) and configuration: every setting in one cell |
| 1 | Data curation → curated dataset + curation log |
| 2 | **Task 1**: natural noise (imbalance, replicate and cross-assay disagreement) |
| 3 | Synthetic noise injection |
| 4 | Models (MLP, GINE, RF) and training, incl. label smoothing and loss correction |
| 5 | Noise detectors (out-of-fold probabilities, confident learning, ensemble filter) |
| 6 | **Tasks 2 + 3**: noise level × defence × model × seed grid |
| 7 | Results: baseline-vs-main comparison table, degradation curves, gain heatmap |
| 8 | **Task 4**: detector flags vs. real suspect labels |
| 9 | **Stretch**: multi-task noise interaction |
| 10 | Model cards, package versions and the output zip |

## How to run (Google Colab, free T4; no Google Drive needed)
1. Open the notebook in Colab (*File → Upload notebook*) and set *Runtime → Change runtime type → T4 GPU*.
2. Run all cells. The Tox21 data is downloaded automatically. Outputs are written to the Colab runtime.
   After each seed of the grid, and at the end, `CM4044_MP2-30_outputs.zip` is downloaded to your computer.
   Allow multiple downloads when the browser asks. The zip contains:
   * the curated dataset and curation log;
   * results tables and figures;
   * model cards and checkpoints.
3. If Colab disconnects, set `RESUME_FROM_ZIP = True` in section 0, run all cells and upload the latest
   zip. Finished runs are skipped.
4. Submit the notebook (*File → Download → .ipynb*, with outputs), the final zip and the slides.

Keep the tab open during long runs: free Colab disconnects after ~90 min without interaction.

**Splitting the grid across the group.** In section 6, each member replaces `SEEDS` with their own
seed (`[0]`, `[1]`, `[2]`) on their own account. To combine, one person sets `RESUME_FROM_ZIP = True`,
selects all members' zips at once in the upload dialog, and runs sections 7–10.

**Quick smoke test.** Set `QUICK = True` in the configuration cell. This runs the whole notebook in
about 10 minutes, with 1 seed, 1 noise level and a few epochs, and writes to separate
`results_quick/` and `figures_quick/` folders.

**Locally (outside Colab):** run `pip install -r requirements.txt`, then run the notebook in Jupyter.
Outputs go to `./mp2_work/`.

### Compute budget
| Step | Colab T4 (estimate) |
|---|---|
| Curation + features | ~1 min |
| Detector out-of-fold probabilities, per noise condition | ~1 min |
| MLP run / RF run / GNN run | ~5 s / ~60 s / ~1 min |
| Full grid: 7 conditions × 3 seeds × (3 RF + 6 MLP + 6 GNN runs) | ≈ 4–5 h in total, resumable |
| Task 4 + stretch | ~40 min |

## Repository layout
```
notebooks/MP2-30_Tox21_label_noise.ipynb   the whole project (code + explanations)
data/raw/tox21.csv.gz                      raw MoleculeNet file (the notebook also downloads it; sha256-checked)
docs/slides_outline.md                     12-slide outline with speaker split and likely questions
requirements.txt                           package versions for a local run
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

**Synthetic noise (Task 2)** is injected into train + valid labels only.
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
