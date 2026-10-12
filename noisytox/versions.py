"""Record the software environment next to the results (reproducibility)."""
import json
import platform

from . import config as C


def main():
    import cleanlab, numpy, pandas, rdkit, scipy, sklearn, torch, torch_geometric
    v = {"python": platform.python_version(), "torch": torch.__version__,
         "cuda": torch.version.cuda if torch.cuda.is_available() else None,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "torch_geometric": torch_geometric.__version__, "rdkit": rdkit.__version__,
         "cleanlab": cleanlab.__version__, "scikit-learn": sklearn.__version__,
         "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "versions.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
