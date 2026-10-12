# Curation log - Tox21

Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz` (MoleculeNet/DeepChem release)  
SHA-256 of raw file: `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360`

| Step | Decision | Records after step |
|---|---|---|
| 0 | Raw file loaded: 7831 records x 12 assays | 7831 |
| 1 | RDKit standardisation: cleanup/normalise, keep largest organic fragment (245 multi-component records, i.e. salts/mixtures), neutralise charges, canonical isomeric SMILES. 835 SMILES changed. Tautomers *not* canonicalised (can change assay-relevant chemistry). Counter-ions are assumed not to drive activity. | 7831 |
| 2 | Dropped records that failed standardisation or are not organic (no carbon / <3 heavy atoms): {"no_carbon": 79, "unparseable": 8, "too_small": 5, "standardisation_failed": 1} | 7738 |
| 3 | Dropped records with no label in any assay: 0 | 7738 |
| 4 | Merged duplicate parent structures: 153 structures had >1 record (216 records removed). Of 1492 (structure, assay) cells measured more than once, 87 (5.8%) have conflicting labels (see `duplicate_conflicts.csv`); 36 resolved by majority, 51 ties set to missing. | 7522 |
| 5 | Deterministic Bemis-Murcko scaffold split 80/10/10 (largest scaffold sets -> train): {"train": 6017, "test": 753, "valid": 752}; 2266 unique scaffolds | 7522 |
| 6 | No subsampling needed: 7522 compounds <= 10k budget | 7522 |

## Label statistics after curation

| Assay | labelled | actives | active rate | missing |
|---|---|---|---|---|
| NR-AR | 6955 | 298 | 0.043 | 567 |
| NR-AR-LBD | 6488 | 228 | 0.035 | 1034 |
| NR-AhR | 6277 | 748 | 0.119 | 1245 |
| NR-Aromatase | 5570 | 283 | 0.051 | 1952 |
| NR-ER | 5947 | 760 | 0.128 | 1575 |
| NR-ER-LBD | 6680 | 334 | 0.050 | 842 |
| NR-PPAR-gamma | 6195 | 174 | 0.028 | 1327 |
| SR-ARE | 5606 | 893 | 0.159 | 1916 |
| SR-ATAD5 | 6793 | 257 | 0.038 | 729 |
| SR-HSE | 6213 | 346 | 0.056 | 1309 |
| SR-MMP | 5579 | 888 | 0.159 | 1943 |
| SR-p53 | 6511 | 408 | 0.063 | 1011 |

## Notes
* Test-set labels are never modified by any noise-injection or cleaning step.
* Synthetic noise (Task 2) is injected into train+valid labels only, using seeded RNGs (see `noisytox/noise.py`).
* Columns: `mol_id` (merged source IDs, `;`-separated), `smiles_raw` (first raw record), `smiles` (standardised parent), 12 assay labels (1/0/blank), `scaffold`, `split`.
