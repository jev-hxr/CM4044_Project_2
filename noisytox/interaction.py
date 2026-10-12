"""Stretch goal - multi-task noise interaction.

Question: in a shared multi-task network, does label noise in some assays
hurt the *other* (clean) assays, and does cleaning the noisy assays help
the clean ones?

Design: the assays in config-like NOISY_TASKS (default: the 7 nuclear-receptor
assays) receive heavy symmetric noise; the 5 stress-response assays stay clean.
Four training regimes, identical everywhere else:
  clean        - no synthetic noise anywhere (upper reference)
  noisy        - NR assays noisy, no defence
  noisy+CL     - NR assays noisy, confident-learning cleaning applied to the NR assays only
  drop_noisy   - NR labels discarded entirely (5-task model on the clean SR assays)
Metrics are reported separately for the clean SR assays and the noisy NR assays.

Run:  python -m noisytox.interaction [--models mlp gnn] [--rate 0.2]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import noise
from .data import Dataset, ecfp4, graphs
from .detectors import oof_probabilities, run_detectors
from .train import NNTrainer, task_metrics

NOISY_TASKS = [t for t in C.TASKS if t.startswith("NR-")]
CLEAN_TASKS = [t for t in C.TASKS if t not in NOISY_TASKS]
REGIMES = ["clean", "noisy", "noisy+CL", "drop_noisy"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mlp", "gnn"], choices=["mlp", "gnn"])
    ap.add_argument("--rate", type=float, default=0.20)
    ap.add_argument("--kind", default="sym", choices=["sym", "cc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    ds = Dataset()
    X = ecfp4(ds.smiles)
    gs = graphs(ds.smiles) if "gnn" in args.models else None
    nz = [C.TASKS.index(t) for t in NOISY_TASKS]
    cl_idx = [C.TASKS.index(t) for t in CLEAN_TASKS]
    dev, te = ds.idx["dev"], ds.idx["test"]
    out_dir = C.RESULTS_DIR / "interaction"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        Yn_all, _ = noise.inject(ds.Y, dev, args.kind, args.rate, seed)
        Yn = ds.Y.copy()
        Yn[:, nz] = Yn_all[:, nz]                   # noise only in the NR assays
        cache = C.RESULTS_DIR / "cache" / f"oof_interaction_{args.kind}_{args.rate:.2f}_s{seed}.npz"
        if cache.exists():
            z = np.load(cache)
            P = {m: z[m] for m in C.DETECTOR_MEMBERS}
        else:
            P = oof_probabilities(X, Yn, dev, seed=seed, device=args.device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **P)
        det = run_detectors(X, Yn, dev, seed=seed, P_members=P)
        labels = {"clean": ds.Y.copy(), "noisy": Yn.copy(), "noisy+CL": Yn.copy(), "drop_noisy": ds.Y.copy()}
        flags = det["cl_flags"].copy()
        flags[:, cl_idx] = False                        # clean only the noisy assays
        labels["noisy+CL"][flags] = np.nan
        labels["drop_noisy"][:, nz] = np.nan
        for model in args.models:
            tr = NNTrainer(model, X=X, graphs=gs, device=args.device)
            for regime in REGIMES:
                path = out_dir / f"{model}_{regime}_{args.kind}_{args.rate:.2f}_s{seed}.json"
                if path.exists():
                    continue
                Ytr = labels[regime]
                # early stopping on the assays the regime actually trains on
                m = tr.fit(Ytr, ds.idx["train"], ds.idx["valid"], seed=seed)
                met = task_metrics(ds.Y[te], tr.predict(m, te))
                rec = dict(model=model, regime=regime, kind=args.kind, rate=args.rate, seed=seed,
                           roc_clean_tasks=float(np.nanmean(met[cl_idx, 0])),
                           pr_clean_tasks=float(np.nanmean(met[cl_idx, 1])),
                           roc_noisy_tasks=float(np.nanmean(met[nz, 0])) if regime != "drop_noisy" else None,
                           pr_noisy_tasks=float(np.nanmean(met[nz, 1])) if regime != "drop_noisy" else None,
                           labels_removed_noisy_tasks=int(flags[:, nz].sum()) if regime == "noisy+CL" else 0)
                path.write_text(json.dumps(rec, indent=1))
                print(f"  {model} {regime:10s} s{seed}  clean-SR ROC {rec['roc_clean_tasks']:.3f}  "
                      f"noisy-NR ROC {rec['roc_noisy_tasks'] or float('nan'):.3f}", flush=True)

    df = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))])
    df.to_csv(C.RESULTS_DIR / "interaction.csv", index=False)
    g = df.groupby(["model", "kind", "rate", "regime"], sort=False)
    summ = g[["roc_clean_tasks", "pr_clean_tasks", "roc_noisy_tasks", "pr_noisy_tasks"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    md = ["# Stretch - multi-task noise interaction\n",
          f"Noisy assays: {', '.join(NOISY_TASKS)}; clean assays: {', '.join(CLEAN_TASKS)}.\n",
          summ.round(4).to_markdown(index=False)]
    (C.RESULTS_DIR / "interaction_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    models = list(dict.fromkeys(df.model))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in ((axes[0], "roc_clean_tasks", "clean SR assays"),
                           (axes[1], "roc_noisy_tasks", "noisy NR assays")):
        for k, model in enumerate(models):
            sub = df[df.model == model].groupby("regime")[col]
            mean = sub.mean().reindex(REGIMES)
            sd = sub.std().reindex(REGIMES).fillna(0)
            ax.bar(np.arange(len(REGIMES)) + (k - 0.5 * (len(models) - 1)) * 0.35, mean, 0.35,
                   yerr=sd, capsize=3, label=model.upper())
        ax.set_xticks(range(len(REGIMES)), REGIMES)
        lo = np.nanmin(df[col].astype(float)) if df[col].notna().any() else 0.5
        ax.set_ylim(max(0.5, lo - 0.05), None)
        ax.set_title(f"test ROC-AUC on {title}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig_interaction.png", dpi=200)


if __name__ == "__main__":
    main()
