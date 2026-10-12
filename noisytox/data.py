"""Loading the curated dataset and computing (cached) features."""
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from . import config as C

RDLogger.DisableLog("rdApp.*")


class Dataset:
    """Curated Tox21: label matrix Y (n x 12, NaN = missing) + split indices."""

    def __init__(self, path=C.CURATED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python -m noisytox.curate` first")
        self.df = pd.read_csv(path)
        self.smiles = self.df["smiles"].tolist()
        self.Y = self.df[C.TASKS].to_numpy(dtype=np.float32)
        split = self.df["split"].to_numpy()
        self.idx = {s: np.where(split == s)[0] for s in ("train", "valid", "test")}
        # rows the model developer "owns" (and whose labels may be noisy)
        self.idx["dev"] = np.concatenate([self.idx["train"], self.idx["valid"]])

    def __len__(self):
        return len(self.smiles)


def ecfp4(smiles, n_bits=C.ECFP_BITS, radius=C.ECFP_RADIUS):
    path = C.CACHE_DIR / f"ecfp{2 * radius}_{n_bits}_{len(smiles)}.npy"
    if path.exists():
        return np.load(path)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), X[i])
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    return X


def graphs(smiles):
    """PyG molecular graphs (OGB-style categorical atom/bond features).

    Each graph carries `row` = its index in the dataset so that a batch can
    look up its (possibly noisy) labels from an external label matrix."""
    from torch_geometric.utils import from_smiles

    path = C.CACHE_DIR / f"graphs_{len(smiles)}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    gs = []
    for i, s in enumerate(smiles):
        g = from_smiles(s)
        g.row = torch.tensor([i])
        del g.smiles
        gs.append(g)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gs, path)
    return gs
