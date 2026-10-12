"""Task 2 - controlled synthetic label noise.

Two noise models, applied independently to every assay, only to *observed*
labels in the developer rows (train + valid).  Test labels are never touched.

* ``sym`` (symmetric / uniform): exactly round(rate * n_labelled) labels,
  drawn uniformly regardless of class, are flipped.  Because ~95 % of Tox21
  labels are inactive, this mostly creates *false actives*: at rate 0.20 the
  noisy active set is dominated by flipped inactives.

* ``cc`` (class-conditional, prior-preserving): a fraction ``rate`` of the
  actives is flipped to inactive (missed actives, e.g. cytotoxicity masking or
  solubility loss) and the *same number* of inactives is flipped to active
  (assay artefacts, e.g. auto-fluorescence / luciferase interference).
  P(1->0) = rate, P(0->1) = rate * pi / (1 - pi).  The class balance looks
  unchanged, so this noise cannot be spotted from label statistics alone.
"""
import numpy as np


def inject(Y, rows, kind, rate, seed):
    """Return (Y_noisy, flipped_mask). Y is n x T with NaN for missing."""
    Yn = Y.copy()
    flipped = np.zeros_like(Y, dtype=bool)
    if kind == "none" or rate == 0:
        return Yn, flipped
    rng = np.random.default_rng(10_000 + seed)
    for t in range(Y.shape[1]):
        y = Y[rows, t]
        obs = rows[~np.isnan(y)]
        if kind == "sym":
            k = int(round(rate * len(obs)))
            pick = rng.choice(obs, size=k, replace=False)
        elif kind == "cc":
            pos = obs[Y[obs, t] == 1]
            neg = obs[Y[obs, t] == 0]
            k = int(round(rate * len(pos)))
            pick = np.concatenate([rng.choice(pos, size=k, replace=False),
                                   rng.choice(neg, size=k, replace=False)])
        else:
            raise ValueError(kind)
        Yn[pick, t] = 1 - Y[pick, t]
        flipped[pick, t] = True
    return Yn, flipped


def condition_name(kind, rate):
    return "clean" if kind == "none" or rate == 0 else f"{kind}_{rate:.2f}"
