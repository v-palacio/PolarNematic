# PolarNematic

Ferronematic liquid crystal dataset with MOL structures, DFT-derived descriptors (COSMO sigma profiles, dipoles) and graph-spectral features, plus scripts to reproduce all calculations from the paper. Molecular structures are literature-sourced; descriptors are computed via ORCA/COSMO and RDKit using the provided `src/` pipelines.

## Layout

```
data/
  datasets/                # CSVs (combined, descriptors)
  dft_results/             # ORCA output data for the lowest energy conformer
    idx_molname/           # Contains geometry.xyz, charges.csv, dft_results.csv
  ml/
    classifier_LR/         # results/plots
    classifier_RF/         # models/results/plots
  molecular_structures/    # .mol files
src/
  graphmol/                # spectrum_mol.py
  ml/                      # classifier_LR.py, classifier_RF.py
  dft/                     # ORCA helpers
  sigma/                   # COSMO-SAC sigma profile tools
  utils/                   # general utilities
```

## Authors

- Dr. Viviana Palacio-Betancur
- Prof. Nick Jackson

University of Illinois at Urbana-Champaign.

## Contact

- Open an issue in this repository for bugs/questions
- For direct inquiries, please email Prof. Jackson (jacksonn@illinois.edu) or Dr. Palacio (vpalacio@illinois.edu)

## License & Citation

- CC-BY 4.0 (see LICENSE)
- Palacio-Betancur V, Jackson N. Molecular Charge Topologies Govern Polar Nematic Ordering. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-crg6c. This content is a preprint and has not been peer-reviewed.
