# graphmol: Molecular Charge Spectrum

Minimal tools to compute and visualize the graph-Fourier spectrum of atomic charges on molecular graphs (normalized Laplacian).

## What it does
- Build a topological Laplacian from an RDKit `Chem.Mol` (normalized, λ ∈ [0, 2]).
- Project per-atom charges onto Laplacian eigenvectors to get a spectrum.
- Plot the spectrum and extract a few simple summary stats.

## API
- `compute_charge_spectrum(mol, charges, center=False, normalize=False)` → `(eigenvalues, squared_coefficients)`
- `build_topological_laplacian(mol, normalized=True)` → `np.ndarray`
- `plot_charge_spectrum(eigenvalues, squared_coefficients, title="...", figsize=(6,4))` → `fig | None`
- `analyze_spectrum_features(eigenvalues, squared_coefficients)` → `dict`

## Quick start
```python
from rdkit import Chem
from graphmol import compute_charge_spectrum, plot_charge_spectrum, analyze_spectrum_features

mol = Chem.MolFromSmiles("c1ccccc1")
charges = [0.0 for _ in range(mol.GetNumAtoms())]  # replace with your charges

ev, spec = compute_charge_spectrum(mol, charges, center=True, normalize=True)
plot_charge_spectrum(ev, spec)
features = analyze_spectrum_features(ev, spec)
```

See `src/graphmol/examples/charge_spectrum_demo.ipynb` for a minimal notebook demo. 