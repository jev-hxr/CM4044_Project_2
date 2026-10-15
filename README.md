# CM4044 Mini Project 2 · MP2-30
## How Noisy Is Toxicity? Label Noise and Ensemble Robustness on Tox21

**Question.** Toxicity labels come from real assays with real error rates. How much label noise can
training pipelines tolerate, and which defences actually work?

**Answer in one line.** About 5 % random label noise is tolerable, but above 10 % every model degrades.
Cleaning noisy labels (ensemble-disagreement filtering) helps the random forest and MLP most, while the
graph neural network is robust on its own. A confident-learning noise detector flags *real* suspect labels
in Tox21 2.4–5.7× more often than consistent ones.

---

## Submission contents

| Deliverable | Where |
|---|---|
| **Code** | `MP2-30_Tox21_label_noise.ipynb`, a single self-contained Google Colab notebook with all code, explanations and outputs |
| **Environment** | `requirements.txt`; exact versions used in the run are recorded in `results/versions.json` |
| **Curated dataset** | `data/tox21_curated.csv` + `data/curation_log.md` (every filtering, standardisation and deduplication decision) + `data/duplicate_conflicts.csv` |
| **Trained models** | `models/checkpoints/`: undefended baselines and defended models (MLP, GIN: `.pt`; RF: `rf_none_clean_s0.joblib`) |
| **Model cards** | `models/cards/`: one card per model–defence pair (15 cards) |
| **Results** | `results/` (tables, incl. `comparison_table.md`) and `figures/` (all plots) |
| **Slides** | 12-slide presentation (separate file) |

---

## How to reproduce

1. Open `MP2-30_Tox21_label_noise.ipynb` in **Google Colab** and select *Runtime → Change runtime type → T4 GPU*.
2. *Runtime → Run all*. The notebook installs the three packages Colab lacks (RDKit, PyTorch Geometric,
   cleanlab), downloads Tox21 automatically (checked against its SHA-256), and runs every step.
   No Google Drive is needed.
3. A progress zip (`CM4044_MP2-30_outputs.zip`) is downloaded after each seed of the experiment grid and
   once at the end. If Colab disconnects, set `RESUME_FROM_ZIP = True` in the first code cell, run all
   cells again and upload the latest zip: finished runs are skipped.

**Run time on a free Colab T4** (settings below): about 1–1.5 h for the grid, plus about 20 min for
the rest. Setting `QUICK = True` in the configuration cell runs a 10-minute smoke test instead; it writes
to separate `results_quick/` and `figures_quick/` folders.

**Running locally:** `pip install -r requirements.txt`, then open the notebook in Jupyter. A GPU is
optional; outputs go to `./mp2_work/`.

### Settings used for the reported results
All settings live in the configuration cell (section 0) and are identical for every model and defence.

| Setting | Value |
|---|---|
| Seeds | 0, 1 |
| Noise types | symmetric (`sym`), class-conditional prior-preserving (`cc`) |
| Noise rates | 5 %, 10 %, 20 % (train + validation labels only; test labels never changed) |
| Split | deterministic Bemis–Murcko scaffold split, 80 / 10 / 10 |
| MLP | ECFP4 (2048 bits) → 1024 → 256 → 12, dropout 0.3, Adam 1e-3, ≤ 80 epochs, patience 12 |
| GIN | GINE, 4 residual layers × 128, mean+max pooling, dropout 0.2, Adam 1e-3, ≤ 40 epochs, patience 6 |
| RF | 50 trees per assay on ECFP4 |
| Detectors | 5-fold out-of-fold probabilities from logistic regression, Tanimoto kNN and an MLP |
| Total grid | 7 conditions × 2 seeds × 15 model–defence pairs = 206 runs |

---

## What the notebook does

| Section | Content |
|---|---|
| 0 | Setup and configuration |
| 1 | **Data curation**: RDKit standardisation (largest organic fragment, neutralisation, canonical SMILES), removal of invalid/inorganic records, merging of duplicate structures (conflicts logged), scaffold split → 7,831 raw records become 7,522 compounds |
| 2 | **Task 1, natural noise**: class imbalance per assay, replicate disagreement, cross-assay disagreement (NR-AR vs NR-AR-LBD, NR-ER vs NR-ER-LBD) → definition of *suspect labels* |
| 3 | **Task 2, synthetic noise**: seeded label flipping, symmetric and class-conditional |
| 4 | Models (RF, multi-task MLP, GINE graph network) and training; label smoothing and forward loss correction live in the loss function |
| 5 | Noise detectors: confident learning (cleanlab) and ensemble-disagreement filtering, both on out-of-fold predictions |
| 6 | **Tasks 2 + 3, experiment grid**: every noise condition × model × defence × seed, scored on the clean test set |
| 7 | Results: baseline-vs-main comparison table, degradation curves, defence-gain heat map, detector precision/recall |
| 8 | **Task 4**: do the detectors flag real suspect labels more than concordant ones? (within label class, Fisher exact test) |
| 9 | **Stretch**: multi-task noise interaction (noise in the 7 NR assays only; effect on the 5 clean SR assays) |
| 10 | RF checkpoint, model cards, package versions and the output zip |

### Baseline vs. main models
- **Baseline:** the *undefended* model (standard training on the noisy labels), for RF, MLP and GIN.
  The RF on ECFP4 is also the classical baseline.
- **Main models:** the same architectures with one defence each, under identical splits, seeds,
  hyperparameters and metrics:

| Defence | Idea | Applied to |
|---|---|---|
| Label smoothing (`ls`) | Targets 0.95 / 0.05 instead of 1 / 0 | MLP, GIN |
| Loss correction (`fc`) | Correct the loss for flip rates estimated by confident learning | MLP, GIN |
| Loss correction, oracle (`fc_oracle`) | Same, with the true injected flip rates (diagnostic only) | MLP, GIN |
| Confident learning (`cl`) | Remove labels the out-of-fold models confidently disagree with | RF, MLP, GIN |
| Ensemble filter (`ed`) | Remove labels that ≥ 2 of 3 out-of-fold models call wrong | RF, MLP, GIN |

Label smoothing and loss correction modify a neural network's training loss, so they do not apply to
random forests.

---

## Key results (mean of 2 seeds, test ROC-AUC, 12-assay mean)

**Task 1, natural noise**
- Actives are only 3–16 % of labels per assay (inactive:active ratios from 1:5 to 1:35).
- 5.8 % of repeated measurements conflict (NR-ER 18.6 %, SR-ARE 16.3 %).
- Related assays agree only moderately (Cohen's kappa 0.57 AR pair, 0.49 ER pair). 426 compounds are
  active only in the full-length ER assay vs 29 only in ER-LBD, consistent with assay artefacts.

**Task 2, degradation of undefended models**

| Model | Clean | Sym 20 % | CC 20 % |
|---|---|---|---|
| RF | 0.680 | 0.621 | 0.657 |
| MLP | 0.698 | 0.629 | 0.684 |
| **GIN** | **0.733** | **0.693** | **0.740** |

Symmetric noise is the most damaging, because it mostly creates false actives. The GIN is the best
and most robust model in every condition.

**Task 3, defences**
- The **ensemble filter** gives the most consistent gains for RF (+0.020 to +0.028 under noise) and MLP.
- **Loss correction** helps the MLP at high symmetric noise (+0.020), but it is limited by the
  noise-rate estimate: the active→inactive flip rate is estimated at about 0.71 when the true rate is 0.05–0.20.
  With the true rates, the GIN gains +0.022.
- **Label smoothing** never helps. No practical defence reliably helps the already-robust GIN.

**Task 4, real suspect labels**
- Confident learning flags 62.7 % of suspect actives vs 26.6 % of concordant actives
  (2.4×, Fisher p = 6 × 10⁻⁴²), and 3.3 % vs 0.6 % of inactives (5.7×, p = 1 × 10⁻⁹).
- The detector sees only chemical structures, never the partner assay's label.

**Stretch.** Noise in the NR assays did not measurably affect the clean SR assays (≤ 0.005), and cleaning
the noisy assays did not help the clean ones.

Full tables: `results/comparison_table.md`, `results/suspects_report.md`, `results/interaction_report.md`.

### Limitations
- Only 2 seeds: differences below about 0.01 ROC-AUC are within seed-to-seed variation.
- A suspect label is not a proven error; disagreeing assays may reflect real biology.
- The synthetic noise is structure-independent, unlike real assay errors.
- ToxCast (optional second noise profile in the brief) was not used.

---

## Reproducibility
- Fixed seeds for noise injection (`10000 + seed`), model initialisation, batch order and cross-validation folds.
- The scaffold split is deterministic, and the raw data file is verified by SHA-256.
- Package versions and GPU are recorded in `results/versions.json`.
- Bit-identical GPU results are not guaranteed (PyTorch Geometric uses atomic operations on CUDA);
  differences are far smaller than the seed-to-seed spread.

## References
1. Mayr, A.; Klambauer, G.; Unterthiner, T.; Hochreiter, S. DeepTox: Toxicity prediction using deep learning. *Front. Environ. Sci.* 2016, 3, 80.
2. Northcutt, C. G.; Jiang, L.; Chuang, I. L. Confident learning: Estimating uncertainty in dataset labels. *J. Artif. Intell. Res.* 2021, 70, 1373–1411.
3. Wu, Z.; Ramsundar, B.; Feinberg, E. N.; et al. MoleculeNet: a benchmark for molecular machine learning. *Chem. Sci.* 2018, 9, 513–530.
4. Patrini, G.; Rozza, A.; Menon, A. K.; Nock, R.; Qu, L. Making deep neural networks robust to label noise: a loss correction approach. *CVPR* 2017.
5. Brodley, C. E.; Friedl, M. A. Identifying mislabeled training data. *J. Artif. Intell. Res.* 1999, 11, 131–167.
6. Hu, W.; Liu, B.; Gomes, J.; et al. Strategies for pre-training graph neural networks. *ICLR* 2020.
