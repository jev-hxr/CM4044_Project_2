"""Collect every run JSON into tables + figures (Tasks 2 and 3).

Outputs
  results/grid.csv                  one row per run (model, defence, condition, seed)
  results/grid_per_task.csv         per-assay metrics
  results/summary.csv               mean / sd over seeds
  results/delta_vs_undefended.csv   paired (same seed) improvement of every defence
  results/detection_summary.csv     detector precision / recall vs injected flips
  results/comparison_table.md       the baseline-vs-main slide table
  figures/fig_degradation_roc.png, fig_degradation_pr.png, fig_delta_heatmap.png,
  figures/fig_detection.png, fig_channel.png

Run:  python -m noisytox.aggregate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

DEF_ORDER = ["none", "ls", "fc", "fc_oracle", "cl", "ed"]
DEF_LABEL = {"none": "undefended (baseline)", "ls": "label smoothing", "fc": "loss correction (est. T)",
             "fc_oracle": "loss correction (oracle T)", "cl": "confident learning",
             "ed": "ensemble-disagreement filter"}
DEF_COLOR = {"none": "#222222", "ls": "#1f77b4", "fc": "#ff7f0e", "fc_oracle": "#ffbb78",
             "cl": "#2ca02c", "ed": "#d62728"}
MODEL_LABEL = {"rf": "RF (ECFP4)", "mlp": "MLP (ECFP4)", "gnn": "GIN (graph)"}
COND_ORDER = ["clean"] + [f"{k}_{r:.2f}" for k in C.NOISE_KINDS for r in C.NOISE_RATES]


def load_runs():
    recs = [json.loads(p.read_text()) for p in sorted((C.RESULTS_DIR / "runs").glob("*.json"))]
    if not recs:
        raise SystemExit("no runs found in results/runs - run `python -m noisytox.run_grid` first")
    df = pd.DataFrame(recs)
    per_task = []
    for r in recs:
        for t in C.TASKS:
            per_task.append(dict(model=r["model"], defence=r["defence"], condition=r["condition"],
                                 seed=r["seed"], task=t, roc_auc=r["roc_auc_tasks"][t],
                                 pr_auc=r["pr_auc_tasks"][t]))
    return df.drop(columns=["roc_auc_tasks", "pr_auc_tasks"]), pd.DataFrame(per_task)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def summarise(df):
    g = df.groupby(["model", "defence", "condition", "kind", "rate"])
    s = g.agg(roc_mean=("roc_auc", "mean"), roc_sd=("roc_auc", "std"),
              pr_mean=("pr_auc", "mean"), pr_sd=("pr_auc", "std"),
              n_seeds=("seed", "nunique"), labels_removed=("labels_removed", "mean")).reset_index()
    base = df[df.defence == "none"][["model", "condition", "seed", "roc_auc", "pr_auc"]]
    d = df.merge(base, on=["model", "condition", "seed"], suffixes=("", "_base"))
    d["d_roc"] = d.roc_auc - d.roc_auc_base
    d["d_pr"] = d.pr_auc - d.pr_auc_base
    delta = d[d.defence != "none"].groupby(["model", "defence", "condition"]).agg(
        d_roc_mean=("d_roc", "mean"), d_roc_sd=("d_roc", "std"),
        d_pr_mean=("d_pr", "mean"), d_pr_sd=("d_pr", "std"),
        wins_roc=("d_roc", lambda x: int((x > 0).sum())), n=("d_roc", "size")).reset_index()
    return s, delta


def comparison_table(s, delta, metric="roc"):
    conds = [c for c in COND_ORDER if c in set(s.condition)]
    head = "| Model | Defence | " + " | ".join(conds) + " |"
    lines = [head, "|" + "---|" * (len(conds) + 2)]
    for model in ["rf", "mlp", "gnn"]:
        for d in DEF_ORDER:
            sub = s[(s.model == model) & (s.defence == d)].set_index("condition")
            if sub.empty:
                continue
            cells = []
            for c in conds:
                if c not in sub.index:
                    cells.append("–")
                    continue
                cell = fmt(sub.loc[c, f"{metric}_mean"], sub.loc[c, f"{metric}_sd"])
                if d != "none":
                    dd = delta[(delta.model == model) & (delta.defence == d) & (delta.condition == c)]
                    if len(dd):
                        v = dd[f"d_{metric}_mean"].iloc[0]
                        cell += f" ({v:+.3f})"
                cells.append(cell)
            name = "**" + DEF_LABEL[d] + "**" if d == "none" else DEF_LABEL[d]
            lines.append(f"| {MODEL_LABEL[model]} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_degradation(s, metric, path):
    models = [m for m in ["rf", "mlp", "gnn"] if m in set(s.model)]
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 7), sharey="row", squeeze=False)
    for i, kind in enumerate(C.NOISE_KINDS):
        for j, model in enumerate(models):
            ax = axes[i, j]
            for d in DEF_ORDER:
                sub = s[(s.model == model) & (s.defence == d) & ((s.kind == kind) | (s.condition == "clean"))]
                if d == "fc_oracle":  # oracle T is identity on clean data
                    sub = sub[sub.condition != "clean"]
                if sub.empty:
                    continue
                sub = sub.sort_values("rate")
                ax.errorbar(sub.rate * 100, sub[f"{metric}_mean"], yerr=sub[f"{metric}_sd"].fillna(0),
                            marker="o", ms=4, capsize=2, lw=2.2 if d == "none" else 1.4,
                            ls="--" if d == "fc_oracle" else "-", color=DEF_COLOR[d], label=DEF_LABEL[d])
            ax.set_title(f"{MODEL_LABEL[model]} - {'symmetric' if kind == 'sym' else 'class-conditional'} noise",
                         fontsize=9)
            ax.set_xlabel("injected noise rate (%)")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"test {'ROC-AUC' if metric == 'roc' else 'PR-AUC'} (mean of 12 assays)")
    axes[0, -1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_delta(delta, path):
    rows = [(m, d) for m in ["rf", "mlp", "gnn"] for d in DEF_ORDER[1:]
            if len(delta[(delta.model == m) & (delta.defence == d)])]
    conds = [c for c in COND_ORDER if c in set(delta.condition)]
    M = np.full((len(rows), len(conds)), np.nan)
    for i, (m, d) in enumerate(rows):
        for j, c in enumerate(conds):
            v = delta[(delta.model == m) & (delta.defence == d) & (delta.condition == c)]
            if len(v):
                M[i, j] = v.d_roc_mean.iloc[0]
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(1.0 * len(conds) + 3, 0.4 * len(rows) + 1.5))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(conds)), conds, rotation=30)
    ax.set_yticks(range(len(rows)), [f"{MODEL_LABEL[m]} | {DEF_LABEL[d]}" for m, d in rows], fontsize=7)
    for i in range(len(rows)):
        for j in range(len(conds)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Δ test ROC-AUC vs undefended (same seed)")
    ax.set_title("Defence gain over the undefended baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def detection_summary():
    recs = []
    for p in sorted((C.RESULTS_DIR / "detection").glob("*.json")):
        r = json.loads(p.read_text())
        est, true = np.array(r["channel_estimated"]), np.array(r["channel_true"])
        for name in ("cl", "ed"):
            d = r[name]
            recs.append(dict(condition=r["condition"], kind=r["kind"], rate=r["rate"], seed=r["seed"],
                             detector=name, flagged=d["flagged"], precision=d["precision"],
                             recall=d["recall"], flag_rate_clean=d["flag_rate_clean"],
                             precision_label1=d["label1_precision"], recall_label1=d["label1_recall"],
                             precision_label0=d["label0_precision"], recall_label0=d["label0_recall"],
                             rho01_est=est[:, 0].mean(), rho10_est=est[:, 1].mean(),
                             rho01_true=true[:, 0].mean(), rho10_true=true[:, 1].mean()))
    return pd.DataFrame(recs)


def plot_detection(det, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    noisy = det[det.kind != "none"]
    g = noisy.groupby(["kind", "rate", "detector"]).mean(numeric_only=True).reset_index()
    for kind, ls in (("sym", "-"), ("cc", "--")):
        for name, col in (("cl", DEF_COLOR["cl"]), ("ed", DEF_COLOR["ed"])):
            sub = g[(g.kind == kind) & (g.detector == name)]
            lab = f"{name.upper()} {kind}"
            axes[0].plot(sub.rate * 100, sub.precision, ls=ls, marker="o", color=col, label=lab)
            axes[1].plot(sub.rate * 100, sub.recall, ls=ls, marker="o", color=col, label=lab)
    axes[0].set_title("Precision of flagged labels (vs injected flips)")
    axes[1].set_title("Recall of injected flips")
    for ax in axes[:2]:
        ax.set_xlabel("injected noise rate (%)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    cl = g[g.detector == "cl"]
    for kind, mk in (("sym", "o"), ("cc", "s")):
        sub = cl[cl.kind == kind]
        axes[2].scatter(sub.rho10_true, sub.rho10_est, marker=mk, color="#9467bd", label=f"ρ(1→0) {kind}")
        axes[2].scatter(sub.rho01_true, sub.rho01_est, marker=mk, color="#8c564b", label=f"ρ(0→1) {kind}")
    axes[2].plot([0, 1], [0, 1], "k:", lw=1)
    axes[2].set_xlim(0, 0.3)
    axes[2].set_xlabel("true flip rate (mean over assays)")
    axes[2].set_ylabel("CL-estimated flip rate")
    axes[2].set_title("Noise-channel estimate used by loss correction")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    C.FIG_DIR.mkdir(exist_ok=True)
    df, per_task = load_runs()
    df.to_csv(C.RESULTS_DIR / "grid.csv", index=False)
    per_task.to_csv(C.RESULTS_DIR / "grid_per_task.csv", index=False)
    s, delta = summarise(df)
    s.to_csv(C.RESULTS_DIR / "summary.csv", index=False)
    delta.to_csv(C.RESULTS_DIR / "delta_vs_undefended.csv", index=False)

    md = ["# Baseline vs. defended models\n",
          "Test-set metrics (clean labels, scaffold split), mean ± sd over seeds; in brackets: paired "
          "mean change vs. the undefended model of the same architecture, condition and seed.\n",
          "## ROC-AUC\n", comparison_table(s, delta, "roc"),
          "\n## PR-AUC\n", comparison_table(s, delta, "pr")]
    det = detection_summary()
    if len(det):
        det.to_csv(C.RESULTS_DIR / "detection_summary.csv", index=False)
        gd = det.groupby(["condition", "detector"])[["flagged", "precision", "recall", "flag_rate_clean",
                                                     "rho01_est", "rho01_true", "rho10_est", "rho10_true"]]
        md += ["\n## Noise detection vs. injected flips (mean over seeds)\n",
               gd.mean().round(3).reset_index().to_markdown(index=False)]
        plot_detection(det, C.FIG_DIR / "fig_detection.png")
    (C.RESULTS_DIR / "comparison_table.md").write_text("\n".join(md) + "\n")
    plot_degradation(s, "roc", C.FIG_DIR / "fig_degradation_roc.png")
    plot_degradation(s, "pr", C.FIG_DIR / "fig_degradation_pr.png")
    plot_delta(delta, C.FIG_DIR / "fig_delta_heatmap.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
