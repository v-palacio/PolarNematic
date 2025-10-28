"""
Graph Fourier Analysis for Molecular Charge Distributions

This module provides tools for analyzing the frequency components of atomic charges
on molecular graphs using graph Fourier transforms.
"""

from .spectrum_mol import (
    build_topological_laplacian,
    compute_charge_spectrum_from_laplacian,
    compute_mol_spectrum as compute_charge_spectrum,
    plot_charge_spectrum,
    analyze_spectrum_features,
)

__version__ = "1.0.0"
__all__ = [
    "build_topological_laplacian",
    "compute_charge_spectrum_from_laplacian",
    "compute_charge_spectrum",
    "plot_charge_spectrum",
    "analyze_spectrum_features",
] 