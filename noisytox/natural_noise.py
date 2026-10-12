"""Task 1 - characterise the *natural* noise in Tox21.

Outputs (results/ and figures/):
  natural_imbalance.csv      labelled / active counts, imbalance ratio per assay
  natural_pairs.csv          agreement statistics for every pair of assays
  natural_related_pairs.csv  contingency tables for the related-assay pairs
  natural_replicates.csv     replicate (duplicate-record) disagreement per assay
  fig_imbalance.png, fig_assay_kappa.png

Run:  python -m noisytox.natural_noise
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import Dataset


def cohen_kappa(a, b):
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def suspect_matrix(Y):
    """n x T boolean: label observed in both assays of a related pair and the two disagree.

    Also returns `paired` (label observed in both assays of a related pair),
    the population against which suspects are compared."""
    suspect = np.zeros_like(Y, dtype=bool)
    paired = np.zeros_like(Y, dtype=bool)
    for a, b in C.RELATED_PAIRS:
        i, j = C.TASKS.index(a), C.TASKS.index(b)
        both = ~np.isnan(Y[:, i]) & ~np.isnan(Y[:, j])
        dis = both & (Y[:, i] != Y[:, j])
        for k in (i, j):
            paired[:, k] |= both
            suspect[:, k] |= dis
    return suspect, paired


def imbalance_table(Y):
    rows = []
    for t, name in enumerate(C.TASKS):
        y = Y[:, t]
        n, pos = int((~np.isnan(y)).sum()), int(np.nansum(y))
        rows.append(dict(assay=name, labelled=n, actives=pos, inactives=n - pos,
                         active_rate=pos / n, imbalance_ratio=(n - pos) / pos,
                         missing_rate=float(np.isnan(y).mean())))
    return pd.DataFrame(rows)


def pair_table(Y):
    rows = []
    for (i, a), (j, b) in itertools.combinations(enumerate(C.TASKS), 2):
        both = ~np.isnan(Y[:, i]) & ~np.isnan(Y[:, j])
        ya, yb = Y[both, i], Y[both, j]
        any_act = (ya == 1) | (yb == 1)
        rows.append(dict(assay_a=a, assay_b=b, n_both=int(both.sum()),
                         both_active=int(((ya == 1) & (yb == 1)).sum()),
                         a_only=int(((ya == 1) & (yb == 0)).sum()),
                         b_only=int(((ya == 0) & (yb == 1)).sum()),
                         kappa=cohen_kappa(ya, yb),
                         jaccard_actives=float(((ya == 1) & (yb == 1)).sum() / max(any_act.sum(), 1)),
                         related=(a, b) in C.RELATED_PAIRS))
    return pd.DataFrame(rows)


def replicate_table():
    """Replicate disagreement (same parent structure, multiple raw records)."""
    from .curate import standardise

    raw = pd.read_csv(C.RAW_PATH)
    raw["parent"] = [standardise(s)[0] for s in raw["smiles"]]
    raw = raw.dropna(subset=["parent"])
    dup = raw[raw.duplicated("parent", keep=False)]
    rows = []
    for t in C.TASKS:
        g = dup.groupby("parent")[t]
        measured = g.count() >= 2
        conflict = g.agg(lambda v: v.dropna().nunique() > 1)[measured]
        rows.append(dict(assay=t, replicated_cells=int(measured.sum()),
                         conflicting=int(conflict.sum()),
                         conflict_rate=float(conflict.mean()) if measured.any() else np.nan))
    return pd.DataFrame(rows)


def main():
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(exist_ok=True)
    ds = Dataset()
    Y = ds.Y

    imb = imbalance_table(Y)
    imb.to_csv(C.RESULTS_DIR / "natural_imbalance.csv", index=False)
    pairs = pair_table(Y)
    pairs.to_csv(C.RESULTS_DIR / "natural_pairs.csv", index=False)
    rep = replicate_table()
    rep.to_csv(C.RESULTS_DIR / "natural_replicates.csv", index=False)

    rel = pairs[pairs.related].copy()
    rel["disagree"] = rel.a_only + rel.b_only
    rel["disagree_rate_all"] = rel.disagree / rel.n_both
    rel["disagree_rate_actives"] = rel.disagree / (rel.disagree + rel.both_active)
    rel.to_csv(C.RESULTS_DIR / "natural_related_pairs.csv", index=False)
    suspect, paired = suspect_matrix(Y)

    print("== Class imbalance ==\n", imb.round(3).to_string(index=False))
    print("\n== Related-assay disagreement ==\n", rel.round(3).to_string(index=False))
    print(f"\nSuspect labels: {suspect.sum()} of {paired.sum()} paired labels")
    print("\n== Replicate disagreement ==\n", rep.round(3).to_string(index=False))
    print(f"overall replicate conflict rate: {rep.conflicting.sum() / rep.replicated_cells.sum():.3f}")
    top = pairs.sort_values("kappa", ascending=False).head(8)
    print("\n== Most concordant assay pairs (kappa) ==\n", top.round(3).to_string(index=False))

    # figures
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(imb.assay, imb.active_rate * 100, color="#3b6ea8")
    for k, v in enumerate(imb.imbalance_ratio):
        ax.text(k, imb.active_rate[k] * 100 + 0.3, f"1:{v:.0f}", ha="center", fontsize=7)
    ax.set_ylabel("actives (% of labelled)")
    ax.set_title("Tox21 class imbalance per assay (label = inactive:active ratio)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_imbalance.png", dpi=200)

    K = np.eye(len(C.TASKS))
    for _, r in pairs.iterrows():
        i, j = C.TASKS.index(r.assay_a), C.TASKS.index(r.assay_b)
        K[i, j] = K[j, i] = r.kappa
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.TASKS)), C.TASKS, rotation=70, fontsize=7)
    ax.set_yticks(range(len(C.TASKS)), C.TASKS, fontsize=7)
    for i in range(len(C.TASKS)):
        for j in range(len(C.TASKS)):
            if i != j:
                ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if K[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Cohen's kappa (co-labelled molecules)")
    ax.set_title("Cross-assay label agreement")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_assay_kappa.png", dpi=200)


if __name__ == "__main__":
    main()
