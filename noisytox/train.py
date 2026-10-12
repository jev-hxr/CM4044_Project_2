"""Training / evaluation utilities shared by every experiment.

All neural models are multi-task (12 sigmoid outputs) with a masked binary
cross-entropy: missing labels (NaN) - including labels removed by a cleaning
defence - contribute nothing to the loss.  Defences plug in here:

* label smoothing   -> targets y*(1-eps) + eps/2
* forward correction (Patrini et al. 2017) -> the clean probability p is
  pushed through the estimated per-task noise channel
  q = p (1 - rho10) + (1 - p) rho01  and the BCE is taken against q.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .models import GIN, MLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------- metrics
def task_metrics(Y, P):
    """Per-task ROC-AUC and PR-AUC (average precision) on observed labels."""
    out = []
    for t in range(Y.shape[1]):
        m = ~np.isnan(Y[:, t])
        y, p = Y[m, t], P[m, t]
        if len(np.unique(y)) < 2:
            out.append((np.nan, np.nan))
        else:
            out.append((roc_auc_score(y, p), average_precision_score(y, p)))
    return np.array(out)  # T x 2


def mean_roc(Y, P):
    return float(np.nanmean(task_metrics(Y, P)[:, 0]))


# ------------------------------------------------------------------------ loss
def masked_bce(logits, y, eps=0.0, T=None):
    mask = ~torch.isnan(y)
    y0 = torch.nan_to_num(y)
    if T is not None:
        p = torch.sigmoid(logits)
        q = (p * (1 - T[:, 1]) + (1 - p) * T[:, 0]).clamp(1e-6, 1 - 1e-6)
        loss = -(y0 * torch.log(q) + (1 - y0) * torch.log(1 - q))
    else:
        target = y0 * (1 - eps) + eps / 2 if eps else y0
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ------------------------------------------------------------------- trainers
class NNTrainer:
    """Train a multi-task MLP (on ECFP4) or GIN (on graphs).

    `fit` early-stops on mean ROC-AUC of the *provided* validation labels -
    in the noise experiments those are noisy too, because a real modeller
    never has clean labels to tune on."""

    def __init__(self, kind, X=None, graphs=None, device="auto", hp=None, n_tasks=len(C.TASKS)):
        self.kind = kind
        self.device = get_device(device)
        self.hp = dict(C.MLP if kind == "mlp" else C.GNN, **(hp or {}))
        self.n_tasks = n_tasks
        if kind == "mlp":
            self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            self.graphs = graphs

    def _new_model(self):
        hp = self.hp
        if self.kind == "mlp":
            return MLP(self.X.shape[1], self.n_tasks, hp["hidden"], hp["dropout"])
        return GIN(self.n_tasks, hp["hidden"], hp["layers"], hp["dropout"])

    def _batches(self, rows, shuffle, gen=None, batch_size=None):
        bs = batch_size or self.hp["batch_size"]
        if self.kind == "mlp":
            order = rows[torch.randperm(len(rows), generator=gen).numpy()] if shuffle else rows
            for i in range(0, len(order), bs):
                r = order[i:i + bs]
                if shuffle and len(r) < 2:  # BatchNorm needs >1 sample
                    continue
                yield self.X[r], torch.as_tensor(r)
        else:
            from torch_geometric.loader import DataLoader
            loader = DataLoader([self.graphs[i] for i in rows], batch_size=bs,
                                shuffle=shuffle, generator=gen, drop_last=False)
            for b in loader:
                if shuffle and b.num_graphs < 2:
                    continue
                yield b.to(self.device), b.row.cpu()

    @torch.no_grad()
    def predict(self, model, rows):
        model.eval()
        out = [torch.sigmoid(model(x)).cpu() for x, _ in self._batches(rows, False, batch_size=512)]
        return torch.cat(out).numpy()

    def fit(self, Y, train_rows, val_rows=None, Yval=None, seed=0, eps=0.0, T=None, epochs=None):
        """Y: full n x T label matrix (NaN = missing) used for training rows."""
        set_seed(seed)
        hp = self.hp
        train_rows = train_rows[~np.isnan(Y[train_rows]).all(axis=1)]
        model = self._new_model().to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        Yt = torch.tensor(Y, dtype=torch.float32)
        Tt = None if T is None else torch.tensor(T, dtype=torch.float32, device=self.device)
        gen = torch.Generator().manual_seed(seed)
        max_epochs = epochs or hp["max_epochs"]
        best, best_state, bad, history = -np.inf, None, 0, []
        for epoch in range(max_epochs):
            model.train()
            for x, r in self._batches(train_rows, True, gen):
                loss = masked_bce(model(x), Yt[r].to(self.device), eps, Tt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if val_rows is None:
                continue
            score = mean_roc((Yval if Yval is not None else Y)[val_rows], self.predict(model, val_rows))
            history.append(score)
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.best_val, model.history = best, history
        return model


def fit_rf(X, Y, rows, seed=0, n_estimators=None):
    """One random forest per task, trained on that task's observed labels."""
    hp = dict(C.RF)
    if n_estimators:
        hp["n_estimators"] = n_estimators
    forests = []
    for t in range(Y.shape[1]):
        r = rows[~np.isnan(Y[rows, t])]
        y = Y[r, t]
        if len(np.unique(y)) < 2:
            forests.append(None)
            continue
        rf = RandomForestClassifier(random_state=seed, **hp)
        rf.fit(X[r], y.astype(int))
        forests.append(rf)
    return forests


def predict_rf(forests, X):
    P = np.full((len(X), len(forests)), np.nan, dtype=np.float32)
    for t, rf in enumerate(forests):
        if rf is not None:
            P[:, t] = rf.predict_proba(X)[:, 1]
    return P
