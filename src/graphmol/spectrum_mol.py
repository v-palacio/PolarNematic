from typing import Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from rdkit import Chem


def build_topological_laplacian(
    mol: Chem.Mol,
    *,
    normalized: bool = True,
) -> np.ndarray:
    """Topological Laplacian from RDKit adjacency (no geometry)."""
    if not isinstance(mol, Chem.Mol):
        raise ValueError("Expected an RDKit Mol for build_topological_laplacian")
    A = Chem.GetAdjacencyMatrix(mol).astype(float)
    d = A.sum(axis=1)
    if not normalized:
        D = np.diag(d)
        return D - A
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-15))
    D_inv_sqrt = np.diag(inv_sqrt)
    I = np.eye(A.shape[0])
    return I - D_inv_sqrt @ A @ D_inv_sqrt


def compute_charge_spectrum_from_laplacian(
    laplacian: Union[np.ndarray, sparse.spmatrix],
    charges: Union[list, np.ndarray],
    center: bool = False,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectrum given Laplacian L and a node signal (charges)."""
    charges_arr = np.asarray(charges, dtype=float)
    L = laplacian.toarray() if sparse.issparse(laplacian) else np.asarray(laplacian, dtype=float)

    if L.shape[0] != L.shape[1]:
        raise ValueError("laplacian must be square")
    if L.shape[0] != charges_arr.shape[0]:
        raise ValueError(f"charges length ({charges_arr.shape[0]}) must equal Laplacian size ({L.shape[0]})")

    if center:
        charges_arr = charges_arr - charges_arr.mean()

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    coefficients = eigenvectors.T @ charges_arr
    spectrum = coefficients**2
    if normalize:
        total = float(np.sum(spectrum))
        if total > 0:
            spectrum = spectrum / total
    return eigenvalues, spectrum


def compute_mol_spectrum(
    mol: Chem.Mol,
    charges: Union[list, np.ndarray],
    center: bool = False,
    normalize: bool = False,
    *,
    normalized_laplacian: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spectrum from molecule and charges using (optionally) normalized Laplacian."""
    L = build_topological_laplacian(mol, normalized=normalized_laplacian)
    return compute_charge_spectrum_from_laplacian(L, charges, center=center, normalize=normalize)


def plot_charge_spectrum(
    eigenvalues: np.ndarray,
    squared_coefficients: np.ndarray,
    title: str = "Molecular Charge Fourier Spectrum",
    figsize: Tuple[int, int] = (6,4),
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
) -> Optional[plt.Figure]:
    """Plot the graph Fourier spectrum; returns fig only if created here."""
    created_new_figure = False
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_new_figure = True
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    ax.stem(eigenvalues, squared_coefficients, basefmt=" ", linefmt='b-', markerfmt='ro')
    ax.set_xlabel('Eigenvalue (lambda)', fontsize=12)
    ax.set_ylabel('|c_k|^2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig if created_new_figure else None


def analyze_spectrum_features(eigenvalues: np.ndarray, squared_coefficients: np.ndarray) -> dict:
    """Return basic summary metrics of a charge spectrum."""
    total_energy = np.sum(squared_coefficients)
    dc_idx = np.argmin(np.abs(eigenvalues))
    dc_component = squared_coefficients[dc_idx]
    dc_fraction = dc_component / total_energy if total_energy > 0 else 0
    high_freq_mask = eigenvalues > 1.0
    high_freq_energy = np.sum(squared_coefficients[high_freq_mask])
    high_freq_fraction = high_freq_energy / total_energy if total_energy > 0 else 0
    if total_energy > 0:
        spectral_centroid = np.sum(eigenvalues * squared_coefficients) / total_energy
    else:
        spectral_centroid = 0
    significant_threshold = 0.05 * np.max(squared_coefficients)
    n_significant = np.sum(squared_coefficients > significant_threshold)
    return {
        'total_energy': total_energy,
        'dc_component': dc_component,
        'dc_fraction': dc_fraction,
        'high_freq_fraction': high_freq_fraction,
        'spectral_centroid': spectral_centroid,
        'n_significant_components': n_significant,
        'eigenvalue_range': (np.min(eigenvalues), np.max(eigenvalues))
    }