"""Generate a model card (markdown) for the baseline and every defended model.

Cards are filled from config.py (architecture / hyperparameters), the curated
data checksum, and - if present - results/summary.csv.

Run:  python -m noisytox.model_cards
"""
import hashlib

import pandas as pd

from . import config as C
from .aggregate import COND_ORDER, DEF_LABEL, MODEL_LABEL

ARCH = {
    "rf": (f"Random forest, one per assay, on {C.ECFP_BITS}-bit ECFP4 (Morgan radius {C.ECFP_RADIUS}) "
           f"fingerprints. Hyperparameters: {C.RF}."),
    "mlp": (f"Multi-task MLP on {C.ECFP_BITS}-bit ECFP4: Linear-BatchNorm-ReLU-Dropout blocks "
            f"{C.MLP['hidden']}, 12 sigmoid outputs, masked BCE (missing labels ignored). "
            f"Adam lr={C.MLP['lr']}, weight decay={C.MLP['weight_decay']}, batch {C.MLP['batch_size']}, "
            f"dropout {C.MLP['dropout']}, <= {C.MLP['max_epochs']} epochs with early stopping "
            f"(patience {C.MLP['patience']}) on mean validation ROC-AUC. ~2.4 M parameters."),
    "gnn": (f"Multi-task GINE graph network (PyTorch Geometric): OGB-style categorical atom/bond "
            f"embeddings, {C.GNN['layers']} residual GINEConv layers of width {C.GNN['hidden']} with "
            f"BatchNorm, mean+max pooling, 2-layer head, 12 sigmoid outputs, masked BCE. "
            f"Adam lr={C.GNN['lr']}, weight decay={C.GNN['weight_decay']}, batch {C.GNN['batch_size']}, "
            f"dropout {C.GNN['dropout']}, <= {C.GNN['max_epochs']} epochs, early stopping "
            f"(patience {C.GNN['patience']}). ~0.34 M parameters."),
}
DEFENCE = {
    "none": "Standard training on the (possibly noisy) labels - the **baseline**.",
    "ls": f"Label smoothing: targets y(1-ε)+ε/2 with ε={C.LABEL_SMOOTHING_EPS}.",
    "fc": ("Forward loss correction (Patrini et al. 2017): the predicted clean probability is passed "
           "through a per-assay noise channel estimated by confident learning (cleanlab "
           "`estimate_latent`) before the BCE."),
    "fc_oracle": ("Forward loss correction with the *true* injected flip rates - an upper bound that "
                  "isolates the effect of noise-channel estimation error. Not deployable in practice."),
    "cl": (f"Confident learning (Northcutt et al. 2021, cleanlab `prune_by_noise_rate`) on {C.OOF_FOLDS}-fold "
           f"out-of-fold probabilities averaged over {C.DETECTOR_MEMBERS}; flagged labels are set to "
           "missing for that assay only, then standard training."),
    "ed": (f"Ensemble-disagreement (majority) filter: each of {C.DETECTOR_MEMBERS} votes a label wrong when its "
           f"out-of-fold probability confidently favours the other class (class-specific thresholds); "
           f">= {C.ED_MIN_VOTES} votes -> label set to missing, then standard training."),
}


def card(model, defence, summary):
    sha = hashlib.sha256(C.CURATED_PATH.read_bytes()).hexdigest()[:16] if C.CURATED_PATH.exists() else "n/a"
    lines = [f"# Model card - {MODEL_LABEL[model]}, {DEF_LABEL[defence]}\n",
             f"**Role:** {'baseline' if defence == 'none' else 'main model (defence against label noise)'}  ",
             "**Project:** CM4044 MP2-30 - label noise and robustness on Tox21\n",
             "## Architecture and training", ARCH[model], "",
             "## Noise handling", DEFENCE[defence], "",
             "## Training data",
             f"`data/tox21_curated.csv` (sha256 prefix `{sha}`), 7,522 standardised parent structures, "
             "12 Tox21 assays, deterministic Bemis-Murcko scaffold split 80/10/10 (see `data/curation_log.md`). "
             "Under a noise condition, train+valid labels are corrupted with `noisytox/noise.py` "
             "(seeded); the test set always keeps the original labels.", "",
             "## Checkpoints",
             f"`models/checkpoints/{model}_{defence}_<condition>_s0.{'joblib' if model == 'rf' else 'pt'}` "
             "(seed 0; `<condition>` = clean, sym_0.05 ... cc_0.20). "
             + ("Load with `torch.load(...)['state_dict']` into `noisytox.models."
                f"{'MLP' if model == 'mlp' else 'GIN'}` using the stored `hp`." if model != "rf" else
                "Load with `joblib.load` -> list of 12 scikit-learn forests (one per assay)."), ""]
    if summary is not None:
        sub = summary[(summary.model == model) & (summary.defence == defence)].set_index("condition")
        if len(sub):
            lines += ["## Test performance (mean ± sd over seeds, clean test labels)",
                      "| condition | ROC-AUC | PR-AUC | labels removed |", "|---|---|---|---|"]
            for c in COND_ORDER:
                if c in sub.index:
                    r = sub.loc[c]
                    lines.append(f"| {c} | {r.roc_mean:.3f} ± {r.roc_sd:.3f} | {r.pr_mean:.3f} ± {r.pr_sd:.3f} "
                                 f"| {r.labels_removed:.0f} |")
            lines.append("")
    lines += ["## Intended use",
              "Teaching / research: studying how label noise affects toxicity classifiers and comparing "
              "defences under matched conditions. Ranking compounds for *in vitro* Tox21-type assay "
              "activity only.", "",
              "## Limitations",
              "* Not a regulatory or safety tool; Tox21 assays are *in vitro* HTS readouts, not *in vivo* toxicity.",
              "* Scaffold-split test set contains rarer chemotypes; per-assay test sets hold only 20-117 actives, "
              "so per-assay metrics are noisy (use the 12-assay mean and seed spread).",
              "* Standardisation keeps the parent fragment only: counter-ion / mixture effects are ignored.",
              "* Applicability domain: small organic molecules similar to the Tox21 10K library.",
              "* Synthetic noise models (uniform / prior-preserving class-conditional) are simplifications "
              "of real assay error, which is structure-dependent."]
    return "\n".join(lines) + "\n"


def main():
    path = C.RESULTS_DIR / "summary.csv"
    summary = pd.read_csv(path) if path.exists() else None
    out = C.MODELS_DIR / "cards"
    out.mkdir(parents=True, exist_ok=True)
    pairs = [("rf", d) for d in ("none", "cl", "ed")]
    pairs += [(m, d) for m in ("mlp", "gnn") for d in ("none", "ls", "fc", "fc_oracle", "cl", "ed")]
    for m, d in pairs:
        (out / f"{m}_{d}.md").write_text(card(m, d, summary))
    print(f"wrote {len(pairs)} model cards to {out}")


if __name__ == "__main__":
    main()
