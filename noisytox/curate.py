"""Task 0 - data curation.

Raw Tox21 (MoleculeNet / DeepChem release) -> standardised, deduplicated,
scaffold-split CSV plus a human-readable curation log.

Run:  python -m noisytox.curate
"""
import hashlib
import json
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config as C

RDLogger.DisableLog("rdApp.*")


def download_raw():
    if not C.RAW_PATH.exists():
        C.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(C.RAW_URL, C.RAW_PATH)
    digest = hashlib.sha256(C.RAW_PATH.read_bytes()).hexdigest()
    if digest != C.RAW_SHA256:
        raise RuntimeError(f"Raw file checksum mismatch: {digest}")
    return digest


_cleanup = rdMolStandardize.CleanupParameters()
_lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()


def standardise(smiles):
    """Return (standardised SMILES, status). status != 'ok' means dropped."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Most failures are hypervalent Al/metal salts; retry without sanitisation
        # so that we can still strip the counter-ion.
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None, "unparseable"
        try:
            mol = _lfc.choose(mol)
            Chem.SanitizeMol(mol)
        except Exception:
            return None, "unparseable"
    try:
        mol = rdMolStandardize.Cleanup(mol, _cleanup)   # normalise functional groups, disconnect metals
        mol = _lfc.choose(mol)                           # keep parent (largest organic) fragment
        mol = _uncharger.uncharge(mol)                   # neutralise where possible
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "standardisation_failed"
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return None, "no_carbon"
    if mol.GetNumHeavyAtoms() < 3:
        return None, "too_small"
    return Chem.MolToSmiles(mol), "ok"


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(scaffolds, fracs=C.SPLIT_FRACS):
    """Deterministic DeepChem-style scaffold split: biggest scaffold sets go to
    train, the remaining (rarer) chemotypes fill valid and test."""
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(scaffolds)
    split = np.empty(n, dtype=object)
    n_train, n_valid = fracs[0] * n, (fracs[0] + fracs[1]) * n
    filled = 0
    for g in ordered:
        if filled + len(g) <= n_train:
            name = "train"
        elif filled + len(g) <= n_valid:
            name = "valid"
        else:
            name = "test"
        split[g] = name
        filled += len(g)
    return split


def merge_duplicates(df):
    """Merge records with identical standardised SMILES.

    Per task: concordant labels are kept; discordant labels with a strict
    majority keep the majority; ties become missing (NaN).  Every discordance
    is written to duplicate_conflicts.csv (it is *natural* label noise:
    the same parent compound tested twice gave different answers)."""
    rows, conflicts = [], []
    for smi, g in df.groupby("smiles", sort=False):
        rec = {"smiles": smi, "mol_id": ";".join(g["mol_id"]),
               "smiles_raw": g["smiles_raw"].iloc[0], "n_records": len(g)}
        for t in C.TASKS:
            vals = g[t].dropna().values
            if len(vals) == 0:
                rec[t] = np.nan
            elif len(set(vals)) == 1:
                rec[t] = vals[0]
            else:
                n_pos, n_neg = int((vals == 1).sum()), int((vals == 0).sum())
                rec[t] = 1.0 if n_pos > n_neg else 0.0 if n_neg > n_pos else np.nan
                conflicts.append({"smiles": smi, "mol_id": rec["mol_id"], "task": t,
                                  "n_active": n_pos, "n_inactive": n_neg,
                                  "resolved": rec[t]})
        rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(conflicts)


def replicate_comparisons(df):
    """Number of (structure, task) cells measured >= 2 times."""
    dup = df[df.duplicated("smiles", keep=False)]
    return int((dup.groupby("smiles")[C.TASKS].count() >= 2).sum().sum())


def main():
    digest = download_raw()
    raw = pd.read_csv(C.RAW_PATH)
    log = [f"# Curation log - Tox21\n",
           f"Source: `{C.RAW_URL}` (MoleculeNet/DeepChem release)  ",
           f"SHA-256 of raw file: `{digest}`\n",
           "| Step | Decision | Records after step |", "|---|---|---|",
           f"| 0 | Raw file loaded: {raw.shape[0]} records x {len(C.TASKS)} assays | {raw.shape[0]} |"]

    raw = raw.rename(columns={"smiles": "smiles_raw"})
    out = raw["smiles_raw"].apply(standardise)
    raw["smiles"] = [o[0] for o in out]
    raw["status"] = [o[1] for o in out]
    status_counts = raw["status"].value_counts().to_dict()
    n_multi = int(raw["smiles_raw"].str.contains(r"\.").sum())
    n_changed = int((raw["status"].eq("ok") & (raw["smiles"] != raw["smiles_raw"])).sum())
    log.append(f"| 1 | RDKit standardisation: cleanup/normalise, keep largest organic fragment "
               f"({n_multi} multi-component records, i.e. salts/mixtures), neutralise charges, "
               f"canonical isomeric SMILES. {n_changed} SMILES changed. "
               f"Tautomers *not* canonicalised (can change assay-relevant chemistry). "
               f"Counter-ions are assumed not to drive activity. | {len(raw)} |")
    dropped = {k: v for k, v in status_counts.items() if k != "ok"}
    log.append(f"| 2 | Dropped records that failed standardisation or are not organic "
               f"(no carbon / <3 heavy atoms): {json.dumps(dropped)} | "
               f"{int(raw['status'].eq('ok').sum())} |")
    ok = raw[raw["status"].eq("ok")].copy()
    no_label = ok[C.TASKS].isna().all(axis=1)
    ok = ok[~no_label]
    log.append(f"| 3 | Dropped records with no label in any assay: {int(no_label.sum())} | {len(ok)} |")

    merged, conflicts = merge_duplicates(ok)
    n_dup_groups = int((merged["n_records"] > 1).sum())
    n_ties = int(conflicts["resolved"].isna().sum()) if len(conflicts) else 0
    n_rep = replicate_comparisons(ok)
    log.append(f"| 4 | Merged duplicate parent structures: {n_dup_groups} structures had >1 record "
               f"({len(ok) - len(merged)} records removed). Of {n_rep} (structure, assay) cells measured "
               f"more than once, {len(conflicts)} ({len(conflicts) / max(n_rep, 1):.1%}) have conflicting labels (see `duplicate_conflicts.csv`); "
               f"{len(conflicts) - n_ties} resolved by majority, {n_ties} ties set to missing. | {len(merged)} |")

    merged["scaffold"] = merged["smiles"].apply(scaffold)
    merged["split"] = scaffold_split(merged["scaffold"].tolist())
    counts = merged["split"].value_counts().to_dict()
    log.append(f"| 5 | Deterministic Bemis-Murcko scaffold split 80/10/10 (largest scaffold sets -> train): "
               f"{json.dumps(counts)}; {merged['scaffold'].nunique()} unique scaffolds | {len(merged)} |")
    log.append(f"| 6 | No subsampling needed: {len(merged)} compounds <= 10k budget | {len(merged)} |")

    log.append("\n## Label statistics after curation\n")
    log.append("| Assay | labelled | actives | active rate | missing |")
    log.append("|---|---|---|---|---|")
    for t in C.TASKS:
        y = merged[t]
        log.append(f"| {t} | {int(y.notna().sum())} | {int((y == 1).sum())} | "
                   f"{(y == 1).sum() / y.notna().sum():.3f} | {int(y.isna().sum())} |")
    log.append("\n## Notes\n"
               "* Test-set labels are never modified by any noise-injection or cleaning step.\n"
               "* Synthetic noise (Task 2) is injected into train+valid labels only, using seeded RNGs "
               "(see `noisytox/noise.py`).\n"
               "* Columns: `mol_id` (merged source IDs, `;`-separated), `smiles_raw` (first raw record), "
               "`smiles` (standardised parent), 12 assay labels (1/0/blank), `scaffold`, `split`.")

    cols = ["mol_id", "smiles_raw", "smiles", "n_records"] + C.TASKS + ["scaffold", "split"]
    merged[cols].to_csv(C.CURATED_PATH, index=False)
    conflicts.to_csv(C.CONFLICTS_PATH, index=False)
    C.CURATION_LOG.write_text("\n".join(log) + "\n")
    print("\n".join(log))


if __name__ == "__main__":
    main()
