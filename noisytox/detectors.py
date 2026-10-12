"""Label-noise detectors used by the cleaning defences (Task 3) and the
real-suspect validation (Task 4).

Both detectors start from *out-of-fold* (OOF) predicted probabilities: every
labelled molecule is scored by models that never saw it, so a model cannot
simply memorise a wrong label.  Three deliberately different learners are
used on ECFP4: L2 logistic regression, Tanimoto k-nearest-neighbours and a
multi-task MLP (a random forest was dropped as a member because it is ~10x
slower on Colab's 2 vCPUs and is already the classical baseline model).

* Confident learning (CL; Northcutt et al. 2021) on the members' *averaged*
  OOF probabilities, via the reference implementation `cleanlab`
  (`prune_by_noise_rate`).  CL also estimates the per-task noise transition
  matrix, which feeds the forward loss-correction defence.
* Ensemble-disagreement filter (ED; majority filter in the spirit of Brodley &
  Friedl 1999): each member votes "this label is wrong" when it confidently
  predicts the other class, using class-specific thresholds (the member's mean
  predicted probability for that class - robust to the 5 % active rate).  A
  label is flagged when >= ED_MIN_VOTES members agree.
"""
import warnings

import numpy as np
from cleanlab.count import compute_confident_joint, estimate_latent
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from . import config as C
from .train import NNTrainer, fit_rf, predict_rf


def tanimoto_knn(Xtr, ytr, Xq, k=10, alpha=1.0):
    """Similarity-weighted kNN P(active), shrunk towards the prior with
    `alpha` pseudo-observations."""
    A = Xq.astype(np.float32)
    B = Xtr.astype(np.float32)
    inter = A @ B.T
    sim = inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)
    k = min(k, len(ytr))
    nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    w = np.take_along_axis(sim, nn, axis=1)
    prior = ytr.mean()
    return ((w * ytr[nn]).sum(1) + alpha * prior) / (w.sum(1) + alpha)


def oof_probabilities(X, Y, rows, seed=0, members=C.DETECTOR_MEMBERS, folds=C.OOF_FOLDS,
                      device="auto"):
    """Return {member: P} with P an n x T matrix of OOF P(active) on `rows`."""
    n, T = Y.shape
    out = {m: np.full((n, T), np.nan, dtype=np.float32) for m in members}
    mlp = NNTrainer("mlp", X=X, device=device) if "mlp" in members else None
    kf = KFold(folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(kf.split(rows)):
        tr_rows, va_rows = rows[tr], rows[va]
        if "lr" in members:
            for t in range(T):
                r = tr_rows[~np.isnan(Y[tr_rows, t])]
                lr = LogisticRegression(C=0.1, max_iter=2000)
                lr.fit(X[r], Y[r, t].astype(int))
                out["lr"][va_rows, t] = lr.predict_proba(X[va_rows])[:, 1]
        if "knn" in members:
            for t in range(T):
                r = tr_rows[~np.isnan(Y[tr_rows, t])]
                out["knn"][va_rows, t] = tanimoto_knn(X[r], Y[r, t], X[va_rows])
        if "rf" in members:
            forests = fit_rf(X, Y, tr_rows, seed=seed + k, n_estimators=C.DETECTOR_RF_TREES)
            out["rf"][va_rows] = predict_rf(forests, X[va_rows])
        if "mlp" in members:
            model = mlp.fit(Y, tr_rows, seed=seed + k, epochs=C.DETECTOR_MLP_EPOCHS)
            out["mlp"][va_rows] = mlp.predict(model, va_rows)
    return out


def confident_learning(Y, rows, P):
    """Returns flags (n x T bool), label-quality score (n x T, low = suspect)
    and the estimated noise channel T_hat (T x 2: rho01, rho10)."""
    n, T = Y.shape
    flags = np.zeros((n, T), dtype=bool)
    quality = np.full((n, T), np.nan, dtype=np.float32)
    channel = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        r = rows[~np.isnan(Y[rows, t])]
        y = Y[r, t].astype(int)
        p = np.clip(P[r, t].astype(np.float64), 1e-6, 1 - 1e-6)
        pp = np.stack([1 - p, p], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flags[r, t] = find_label_issues(y, pp, filter_by="prune_by_noise_rate")
            cj = compute_confident_joint(y, pp)
            _, noise_matrix, _ = estimate_latent(cj, y)  # noise_matrix[i, j] = P(label=i | true=j)
        quality[r, t] = pp[np.arange(len(y)), y]
        channel[t] = noise_matrix[1, 0], noise_matrix[0, 1]
    return flags, quality, channel


def ensemble_disagreement(Y, rows, P_members, min_votes=C.ED_MIN_VOTES):
    """Returns flags (n x T bool) and vote fraction (n x T)."""
    n, T = Y.shape
    votes = np.zeros((n, T), dtype=np.float32)
    for P in P_members.values():
        for t in range(T):
            r = rows[~np.isnan(Y[rows, t])]
            y, p = Y[r, t], P[r, t]
            t_pos = p[y == 1].mean()          # typical P(active) of a labelled active
            t_neg = (1 - p[y == 0]).mean()    # typical P(inactive) of a labelled inactive
            wrong = np.where(y == 1, (1 - p) >= t_neg, p >= t_pos)
            votes[r, t] += wrong
    flags = votes >= min_votes
    frac = votes / len(P_members)
    frac[np.isnan(Y)] = np.nan
    return flags, frac


def run_detectors(X, Y, rows, seed=0, device="auto", P_members=None):
    """Convenience wrapper: OOF -> CL + ED.  Returns a dict of results."""
    if P_members is None:
        P_members = oof_probabilities(X, Y, rows, seed=seed, device=device)
    P_avg = np.mean(np.stack(list(P_members.values())), axis=0)
    cl_flags, cl_quality, channel = confident_learning(Y, rows, P_avg)
    ed_flags, ed_votes = ensemble_disagreement(Y, rows, P_members)
    return dict(P_members=P_members, P_avg=P_avg, cl_flags=cl_flags, cl_score=1 - cl_quality,
                channel=channel, ed_flags=ed_flags, ed_score=ed_votes)


def remove_flagged(Y, flags):
    Yc = Y.copy()
    Yc[flags] = np.nan
    return Yc
