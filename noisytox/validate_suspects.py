"""Task 4 - do the noise detectors red-flag *real* suspect labels?

Real suspects = labels of molecules whose two related-assay readouts disagree
(AR full-length vs AR-LBD, ER full-length vs ER-LBD; see config.RELATED_PAIRS).
The detectors (CL and ED, built from out-of-fold predictions on the natural,
un-noised labels of all 7.5k molecules) never see the partner assay's label
of a held-out molecule, so any enrichment is not leakage: it means the label
does not fit the structure-activity pattern learned from *other* molecules.

Comparison population = labels on the four paired assays where both readouts
exist.  Because suspects are strongly enriched in actives and detectors flag
actives more often, everything is also reported *within* label class
(suspect actives vs concordant actives, suspect inactives vs concordant
inactives) - the class-matched comparison is the one to quote.

Outputs: results/suspects_summary.csv, results/suspects_direction.csv,
         results/suspects_report.md, figures/fig_suspects.png

Run:  python -m noisytox.validate_suspects [--seeds 0 1 2]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

from . import config as C
from .data import Dataset, ecfp4
from .detectors import oof_probabilities, run_detectors
from .natural_noise import suspect_matrix

DETECTORS = {"cl": "confident learning", "ed": "ensemble disagreement"}


def compare(flags, score, suspect, population):
    s, c = population & suspect, population & ~suspect
    a, b = int((flags & s).sum()), int((~flags & s).sum())     # suspects flagged / not
    cc, d = int((flags & c).sum()), int((~flags & c).sum())    # concordant flagged / not
    odds, p = fisher_exact([[a, b], [cc, d]], alternative="greater")
    y = np.concatenate([np.ones(s.sum()), np.zeros(c.sum())])
    sc = np.concatenate([score[s], score[c]])
    auc = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else np.nan
    flagged = a + cc
    return dict(n_suspect=a + b, n_concordant=cc + d,
                flag_rate_suspect=a / max(a + b, 1), flag_rate_concordant=cc / max(cc + d, 1),
                lift=(a / max(a + b, 1)) / max(cc / max(cc + d, 1), 1e-9), odds_ratio=odds,
                fisher_p=p, precision=a / max(flagged, 1), recall=a / max(a + b, 1),
                base_rate=(a + b) / max(a + b + cc + d, 1), score_auc=auc)


def direction_table(Y, flags):
    """In disagreeing pairs, which of the two labels does the detector blame?"""
    rows = []
    for a, b in C.RELATED_PAIRS:
        i, j = C.TASKS.index(a), C.TASKS.index(b)
        both = ~np.isnan(Y[:, i]) & ~np.isnan(Y[:, j])
        for act, inact in ((i, j), (j, i)):
            m = both & (Y[:, act] == 1) & (Y[:, inact] == 0)
            rows.append(dict(pair=f"{a} / {b}", active_in=C.TASKS[act], inactive_in=C.TASKS[inact],
                             n=int(m.sum()),
                             active_label_flagged=float(flags[m, act].mean()) if m.any() else np.nan,
                             inactive_label_flagged=float(flags[m, inact].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    ds = Dataset()
    X = ecfp4(ds.smiles)
    Y = ds.Y
    rows = np.arange(len(ds))
    suspect, paired = suspect_matrix(Y)
    strata = {"all paired labels": paired,
              "actives only": paired & (np.nan_to_num(Y, nan=-1) == 1),
              "inactives only": paired & (np.nan_to_num(Y, nan=-1) == 0)}
    for a, b in C.RELATED_PAIRS:
        for t in (a, b):
            m = np.zeros_like(paired)
            m[:, C.TASKS.index(t)] = True
            strata[f"{t} (all)"] = paired & m

    recs, dirs = [], []
    for seed in args.seeds:
        cache = C.RESULTS_DIR / "cache" / f"oof_natural_all_s{seed}.npz"
        if cache.exists():
            z = np.load(cache)
            P = {m: z[m] for m in C.DETECTOR_MEMBERS}
        else:
            P = oof_probabilities(X, Y, rows, seed=seed, device=args.device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **P)
        det = run_detectors(X, Y, rows, seed=seed, P_members=P)
        for key, name in DETECTORS.items():
            flags, score = det[f"{key}_flags"], np.nan_to_num(det[f"{key}_score"])
            for sname, pop in strata.items():
                recs.append(dict(seed=seed, detector=key, stratum=sname, **compare(flags, score, suspect, pop)))
            d = direction_table(Y, flags)
            d.insert(0, "detector", key)
            d.insert(0, "seed", seed)
            dirs.append(d)

    res = pd.DataFrame(recs)
    res.to_csv(C.RESULTS_DIR / "suspects_per_seed.csv", index=False)
    summ = res.groupby(["detector", "stratum"], sort=False).agg(
        n_suspect=("n_suspect", "first"), n_concordant=("n_concordant", "first"),
        flag_rate_suspect=("flag_rate_suspect", "mean"), flag_rate_concordant=("flag_rate_concordant", "mean"),
        lift=("lift", "mean"), odds_ratio=("odds_ratio", "mean"), fisher_p_max=("fisher_p", "max"),
        precision=("precision", "mean"), recall=("recall", "mean"), base_rate=("base_rate", "first"),
        score_auc=("score_auc", "mean"), score_auc_sd=("score_auc", "std")).reset_index()
    summ.to_csv(C.RESULTS_DIR / "suspects_summary.csv", index=False)
    dr = pd.concat(dirs).groupby(["detector", "pair", "active_in", "inactive_in"], sort=False).mean(
        numeric_only=True).drop(columns="seed").reset_index()
    dr.to_csv(C.RESULTS_DIR / "suspects_direction.csv", index=False)

    md = ["# Task 4 - detector flags vs. real (assay-disagreement) suspects\n",
          f"Seeds: {args.seeds}. Suspect = related-assay readouts disagree; concordant = both readouts agree. "
          "`fisher_p_max` = largest one-sided Fisher p over seeds (H1: suspects flagged more often). "
          "`score_auc` = ROC-AUC of the continuous detector score for separating suspect from concordant labels "
          "(0.5 = no signal).\n",
          summ.round(4).assign(fisher_p_max=summ.fisher_p_max.map(lambda v: f"{v:.1e}")).to_markdown(index=False),
          "\n## Which label of a disagreeing pair is flagged?\n", dr.round(3).to_markdown(index=False)]
    (C.RESULTS_DIR / "suspects_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # figure: class-matched flag rates
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    per = res[res.stratum.isin(["all paired labels", "actives only", "inactives only"])]
    for ax, (key, name) in zip(axes, DETECTORS.items()):
        sub = per[per.detector == key].groupby("stratum", sort=False)
        m, s = sub.mean(numeric_only=True), sub.std(numeric_only=True).fillna(0)
        x = np.arange(len(m))
        ax.bar(x - 0.2, m.flag_rate_suspect * 100, 0.4, yerr=s.flag_rate_suspect * 100, capsize=3,
               color="#d62728", label="suspect (related assays disagree)")
        ax.bar(x + 0.2, m.flag_rate_concordant * 100, 0.4, yerr=s.flag_rate_concordant * 100, capsize=3,
               color="#7f7f7f", label="concordant (related assays agree)")
        for k, (lift, p) in enumerate(zip(m.lift, sub.fisher_p.max())):
            ax.text(k, max(m.flag_rate_suspect.iloc[k], m.flag_rate_concordant.iloc[k]) * 100 + 2,
                    f"lift {lift:.1f}x\n" + (f"p={p:.0e}" if p > 0 else "p<1e-300"), ha="center", fontsize=7)
        ax.set_xticks(x, m.index)
        ax.set_title(f"{name}")
        ax.set_ylabel("labels flagged as noisy (%)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylim(0, per.groupby(["detector", "stratum"]).flag_rate_suspect.mean().max() * 100 * 1.3)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Do detectors flag real suspect labels more often than concordant ones?", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_suspects.png", dpi=200)


if __name__ == "__main__":
    main()
