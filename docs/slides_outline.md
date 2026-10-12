# Slide outline: MP2-30 (12 min talk + 3 min questions)

Each slide lists its key figure or table and the file that produces it. Fill in the numbers from
the Colab run. Values marked ⟨…⟩ come from `results/`.

| # | Title | Content | Source |
|---|---|---|---|
| 1 | **How noisy is toxicity?** | Title, group, topic MP2-30, one-sentence question | – |
| 2 | Why label noise matters in toxicity | Tox21 = HTS readouts with real error sources: cytotoxicity masking, luciferase / autofluorescence artefacts, solubility, replicate variability. Models trained on these labels inherit the errors. The question: how much noise can a pipeline tolerate, and which defences work? | DeepTox, MoleculeNet |
| 3 | Data and curation | 7,831 → 7,522 compounds: salt stripping, neutralisation, 79 inorganics dropped, 153 duplicate parents merged. Scaffold split 80/10/10. Every step is in the curation log. | `data/curation_log.md` |
| 4 | Natural noise I: imbalance and replicates | 1:5 to 1:35 inactive:active ratios. 5.8 % of replicate measurements disagree (NR-ER 18.6 %, SR-ARE 16.3 %). | `fig_imbalance.png`, `results/natural_replicates.csv` |
| 5 | Natural noise II: related assays disagree | AR vs AR-LBD κ = 0.57, ER vs ER-LBD κ = 0.49. 426 ER-full-length-only actives vs 29 LBD-only actives. These 1,320 disagreeing labels are the "suspect labels". | `fig_assay_kappa.png`, `results/natural_related_pairs.csv` |
| 6 | Experimental design | Noise models (sym vs prior-preserving cc, 5/10/20 %). Train+valid noisy, test clean. Models: RF, MLP, GIN. Defences: LS, FC (estimated / oracle), CL, ED. 3 seeds, identical splits and hyperparameters. Small pipeline diagram. | README → Methods |
| 7 | Baseline degradation | Undefended RF / MLP / GIN: ROC-AUC and PR-AUC vs noise rate. Symmetric noise hurts most because it floods the rare active class with false positives. | `fig_degradation_roc.png` (black lines) |
| 8 | **Baseline vs. main models (required table)** | ROC-AUC and PR-AUC, mean ± sd, with Δ vs undefended, for clean / sym 10 % / sym 20 % / cc 20 %. Keep 4 columns so it stays readable. | `results/comparison_table.md` |
| 9 | Which defence works, and why | Δ heatmap. Detection precision/recall vs injected flips. Loss correction is limited by noise-channel estimation (estimated vs oracle T). | `fig_delta_heatmap.png`, `fig_detection.png` |
| 10 | **Do detectors catch real suspect labels?** | Flag rate of suspect vs concordant labels, class-matched, with lift, Fisher p and score AUC. Which label of a disagreeing pair gets blamed (e.g. ER full-length actives, which are likely luciferase or cytotoxicity artefacts). | `fig_suspects.png`, `results/suspects_report.md` |
| 11 | Stretch: multi-task noise interaction | Noise in the NR heads: does it hurt the clean SR heads? Does cleaning the NR labels help them? | `fig_interaction.png`, `results/interaction_report.md` |
| 12 | Conclusions and limitations | Tolerable noise level. Best defence per noise type. Detector validated on real suspects (or not). Limitations: synthetic noise is structure-independent; suspects ≠ proven errors; small per-assay test sets; scaffold split. Future work: ToxCast as a second noise profile. | – |

**Speaker split (3 people, about 4 min each):** A = slides 1–5 (data and natural noise), B = 6–9 (grid
and defences), C = 10–12 (real suspects, stretch, conclusions).

**Likely questions to prepare for**
* Why is the validation set noisy? (The realistic setting: a clean validation set would be an oracle.)
* Isn't a suspect just genuine biology (LBD vs full-length receptor)? (Partly. That is why suspects are
  "suspect", not "wrong", and why we compare within label class.)
* Why does symmetric noise hurt so much more than class-conditional noise at the same rate? (Count
  the flipped labels relative to the number of actives.)
* Could the detector be cheating via the partner assay? (No. Probabilities are out-of-fold, so the
  held-out molecule's other labels are never seen.)
