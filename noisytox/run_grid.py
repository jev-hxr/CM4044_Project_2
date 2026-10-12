"""Tasks 2 + 3 - the noise-level x defence x model x seed grid.

For every (noise kind, rate, seed):
  1. inject synthetic noise into the train+valid labels (test stays clean);
  2. compute out-of-fold detector probabilities once (cached) and derive the
     CL / ED flags + the estimated noise channel;
  3. train every (model, defence) pair on identical rows, seeds and
     hyperparameters and score it on the clean test set.

Every run is written to its own JSON file, so the grid is resumable after a
Colab disconnect and can be split across group members (e.g. one seed each)
and merged afterwards with `python -m noisytox.aggregate`.

Examples
  python -m noisytox.run_grid                                   # full default grid
  python -m noisytox.run_grid --models mlp --seeds 0            # quick look
  python -m noisytox.run_grid --models gnn --defences none cl   # subset
"""
import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import noise
from .data import Dataset, ecfp4, graphs
from .detectors import oof_probabilities, remove_flagged, run_detectors
from .train import NNTrainer, fit_rf, predict_rf, task_metrics

DEFENCES = ["none", "ls", "fc", "fc_oracle", "cl", "ed"]
RF_DEFENCES = {"none", "cl", "ed"}  # smoothing / loss correction do not apply to RF


def oracle_channel(Y_clean, flipped, rows):
    """Empirical per-task flip rates actually injected: (rho01, rho10)."""
    T = np.zeros((Y_clean.shape[1], 2), dtype=np.float32)
    for t in range(Y_clean.shape[1]):
        y, f = Y_clean[rows, t], flipped[rows, t]
        T[t] = f[y == 0].mean(), f[y == 1].mean()
    return T


def detection_report(det, flipped, Y_clean, rows, oracle):
    """Precision / recall of each detector against the injected flips."""
    obs = np.zeros_like(flipped)
    obs[rows] = ~np.isnan(Y_clean[rows])
    rep = {"n_labels": int(obs.sum()), "n_flipped": int(flipped.sum())}
    for name in ("cl", "ed"):
        f = det[f"{name}_flags"] & obs
        tp = int((f & flipped).sum())
        rep[name] = dict(flagged=int(f.sum()), tp=tp,
                         precision=tp / max(int(f.sum()), 1),
                         recall=tp / max(int(flipped.sum()), 1) if flipped.any() else float("nan"),
                         flag_rate_clean=float(f[obs & ~flipped].mean()))
        for cls in (0, 1):  # by *noisy* label class
            m = obs & np.nan_to_num(det["Y_noisy"] == cls)
            fm = f & m
            rep[name][f"label{cls}_flagged"] = int(fm.sum())
            rep[name][f"label{cls}_precision"] = float((fm & flipped).sum() / max(fm.sum(), 1))
            rep[name][f"label{cls}_recall"] = float((fm & flipped).sum() / max((m & flipped).sum(), 1))
    rep["channel_estimated"] = det["channel"].tolist()
    rep["channel_true"] = oracle.tolist()
    return rep


def prepare_condition(ds, X, kind, rate, seed, device):
    cond = noise.condition_name(kind, rate)
    Yn, flipped = noise.inject(ds.Y, ds.idx["dev"], kind, rate, seed)
    cache = C.RESULTS_DIR / "cache" / f"oof_{cond}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        P_members = {m: z[m] for m in C.DETECTOR_MEMBERS}
    else:
        t0 = time.time()
        P_members = oof_probabilities(X, Yn, ds.idx["dev"], seed=seed, device=device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **P_members)
        print(f"  [{cond} s{seed}] OOF detector probabilities: {time.time() - t0:.0f}s", flush=True)
    det = run_detectors(X, Yn, ds.idx["dev"], seed=seed, P_members=P_members)
    det["Y_noisy"] = Yn
    oracle = oracle_channel(ds.Y, flipped, ds.idx["dev"])
    rep = detection_report(det, flipped, ds.Y, ds.idx["dev"], oracle)
    rep.update(condition=cond, kind=kind, rate=rate, seed=seed)
    out = C.RESULTS_DIR / "detection" / f"{cond}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    return cond, Yn, det, oracle


def train_labels(defence, Yn, det, oracle):
    """Map a defence to (training labels, smoothing eps, noise channel)."""
    if defence == "none":
        return Yn, 0.0, None
    if defence == "ls":
        return Yn, C.LABEL_SMOOTHING_EPS, None
    if defence == "fc":
        return Yn, 0.0, det["channel"]
    if defence == "fc_oracle":
        return Yn, 0.0, oracle
    if defence == "cl":
        return remove_flagged(Yn, det["cl_flags"]), 0.0, None
    if defence == "ed":
        return remove_flagged(Yn, det["ed_flags"]), 0.0, None
    raise ValueError(defence)


def run(args):
    ds = Dataset()
    X = ecfp4(ds.smiles)
    gs = graphs(ds.smiles) if "gnn" in args.models else None
    trainers = {}
    if "mlp" in args.models:
        trainers["mlp"] = NNTrainer("mlp", X=X, device=args.device)
    if "gnn" in args.models:
        trainers["gnn"] = NNTrainer("gnn", graphs=gs, device=args.device)

    conditions = [] if args.no_clean else [("none", 0.0)]
    conditions += [(k, r) for k in args.kinds for r in args.rates]
    te = ds.idx["test"]
    for seed in args.seeds:
        for kind, rate in conditions:
            cond = noise.condition_name(kind, rate)
            todo = [(m, d) for m in args.models for d in args.defences
                    if not (m == "rf" and d not in RF_DEFENCES)
                    and not (d == "fc_oracle" and cond == "clean")
                    and not (C.RESULTS_DIR / "runs" / f"{m}_{d}_{cond}_s{seed}.json").exists()]
            if not todo and (C.RESULTS_DIR / "detection" / f"{cond}_s{seed}.json").exists():
                continue
            cond, Yn, det, oracle = prepare_condition(ds, X, kind, rate, seed, args.device)
            for model_name, defence in todo:
                t0 = time.time()
                Ytr, eps, T = train_labels(defence, Yn, det, oracle)
                info = {}
                if model_name == "rf":
                    forests = fit_rf(X, Ytr, ds.idx["train"], seed=seed)
                    P = predict_rf(forests, X[te])
                    if args.save_models and seed == 0 and cond == "clean":
                        import joblib
                        path = C.MODELS_DIR / "checkpoints" / f"rf_{defence}_{cond}_s{seed}.joblib"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(forests, path, compress=3)
                else:
                    tr = trainers[model_name]
                    model = tr.fit(Ytr, ds.idx["train"], ds.idx["valid"], seed=seed, eps=eps, T=T)
                    P = tr.predict(model, te)
                    info = dict(epochs=len(model.history), best_val_noisy_roc=model.best_val)
                    if args.save_models and seed == 0:
                        path = C.MODELS_DIR / "checkpoints" / f"{model_name}_{defence}_{cond}_s{seed}.pt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"state_dict": model.state_dict(), "hp": tr.hp,
                                    "kind": model_name, "tasks": C.TASKS}, path)
                m = task_metrics(ds.Y[te], P)
                dev = ds.idx["dev"]
                rec = dict(model=model_name, defence=defence, condition=cond, kind=kind, rate=rate,
                           seed=seed, roc_auc=float(np.nanmean(m[:, 0])), pr_auc=float(np.nanmean(m[:, 1])),
                           roc_auc_tasks=dict(zip(C.TASKS, m[:, 0].tolist())),
                           pr_auc_tasks=dict(zip(C.TASKS, m[:, 1].tolist())),
                           labels_removed=int((~np.isnan(Yn[dev])).sum() - (~np.isnan(Ytr[dev])).sum()),
                           seconds=round(time.time() - t0, 1), **info)
                out = C.RESULTS_DIR / "runs" / f"{model_name}_{defence}_{cond}_s{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rec, indent=1))
                print(f"  {model_name:3s} {defence:9s} {cond:9s} s{seed}  ROC {rec['roc_auc']:.3f}  "
                      f"PR {rec['pr_auc']:.3f}  removed {rec['labels_removed']:5d}  "
                      f"({rec['seconds']:.0f}s)", flush=True)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["rf", "mlp", "gnn"], choices=["rf", "mlp", "gnn"])
    p.add_argument("--defences", nargs="+", default=DEFENCES, choices=DEFENCES)
    p.add_argument("--kinds", nargs="+", default=C.NOISE_KINDS, choices=["sym", "cc"])
    p.add_argument("--rates", nargs="+", type=float, default=C.NOISE_RATES)
    p.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    p.add_argument("--no-clean", action="store_true", help="skip the noise-free condition")
    p.add_argument("--device", default="auto")
    p.add_argument("--save-models", action="store_true", help="save seed-0 checkpoints to models/checkpoints")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
