#!/usr/bin/env python3
"""Compute PCA for a descriptor family and write scores/loadings/bundle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump


_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))

DATA_PATH = os.path.join(_REPO_ROOT, "data", "datasets", "combined_dataset.csv")
OUT_ROOT = os.path.join(_REPO_ROOT, "data", "pca")


# Minimal mapping aligned with classifier_RF expectations
FAMILY_TO_COLS: Dict[str, List[str]] = {
    "sigma_profile": [
        # full sigmap grid columns live in descriptors_sigma.csv, but in combined they appear as stats
        # for PCA users often target the 60-dim sigmap grid; if present in combined, they should be named sigmap_000..059
        # fallback: use any column starting with 'sigmap_'
    ],
    "graphmol_boltzmann": [
        # fallback to columns starting with 'spectrum_'
    ],
}


def _auto_family_columns(df: pd.DataFrame, family: str) -> List[str]:
    if family == "sigma_profile":
        cols = [c for c in df.columns if c.startswith("sigmap_")]
        if not cols:
            # fallback to sigma stats if grid is not present
            cols = [
                c for c in df.columns
                if c.startswith("sigma_") and not c.startswith("spectrum_")
            ]
        return cols
    if family == "graphmol_boltzmann":
        return [c for c in df.columns if c.startswith("spectrum_")]
    raise ValueError(f"Unknown family: {family}")


@dataclass
class PCAConfig:
    data_path: str = DATA_PATH
    out_root: str = OUT_ROOT
    family: str = "graphmol_boltzmann"
    n_components: Optional[int] = None  # None => all
    standardize: bool = False


def run_pca(cfg: PCAConfig) -> None:
    os.makedirs(cfg.out_root, exist_ok=True)
    df = pd.read_csv(cfg.data_path)

    id_col = "Name" if "Name" in df.columns else df.columns[0]
    cols = _auto_family_columns(df, cfg.family)
    if not cols:
        raise ValueError(f"No columns found for family {cfg.family} in {cfg.data_path}")

    X = df[cols].to_numpy(dtype=float)
    # impute
    X = SimpleImputer(strategy="median").fit_transform(X)
    # standardize if requested
    if cfg.standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=cfg.n_components, svd_solver="auto", random_state=42)
    scores = pca.fit_transform(X)

    family_dir = os.path.join(cfg.out_root, cfg.family)
    os.makedirs(family_dir, exist_ok=True)

    # Save scores
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    scores_df.insert(0, "Name", df[id_col].astype(str).values)
    scores_df.to_csv(os.path.join(family_dir, "scores.csv"), index=False)

    # Save loadings
    loadings = pd.DataFrame(pca.components_.T, index=cols, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    loadings.to_csv(os.path.join(family_dir, "loadings.csv"))

    # Save bundle
    bundle = {
        "pca": pca,
        "feature_names": cols,
        "family": cfg.family,
        "standardize": cfg.standardize,
        "data_path": cfg.data_path,
    }
    dump(bundle, os.path.join(family_dir, "bundle.joblib"))

    print(f"PCA saved to {family_dir}")


pass  # no CLI integration


