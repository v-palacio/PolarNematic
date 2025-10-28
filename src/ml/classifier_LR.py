"""Evaluate descriptor families (logistic models) and write CSV/JSON results."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


# Global reproducibility
np.random.seed(42)


# ------------------------------
# Configuration (editable)
# ------------------------------

# Default dataset path: try repository-level merged descriptors if present
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

DATA_PATH: str = os.path.abspath(
    os.path.join(_REPO_ROOT, "data", "full_dataset.csv")
)

# Output directory: output_ml/classifier_LR/results
OUT_DIR: str = os.path.abspath(
    os.path.join(_REPO_ROOT, "data", "ml", "classifier_LR", "results")
)

# Descriptor families mapping (EDITABLE)
# Provide lists of column names belonging to each descriptor family.
# If a family has an empty list or resolves to no valid columns after cleaning,
# it will be skipped and noted in metadata.
FAMILY_TO_COLS: Dict[str, List[str]] = {    
    # Examples (replace with your dataset-specific columns):    
    "dipole_prop": [
      "dipole_Boltzmann_average"
    ],
    "sigma_profile": [
      "sigma_boltzmann_abs_mean",
      "sigma_boltzmann_abs_std",
      "sigma_boltzmann_abs_skewness",
      "sigma_boltzmann_abs_max_val",
      "sigma_boltzmann_abs_p95",
      "sigma_boltzmann_abs_peak_to_mean",
      "sigma_boltzmann_pos_mean",
      "sigma_boltzmann_pos_std",
      "sigma_boltzmann_pos_skewness",
      "sigma_boltzmann_shape_mean",
      "sigma_boltzmann_shape_std",
      "sigma_boltzmann_shape_skewness"
    ],
    "graphmol_conf0": [
      "spectrum_conf0_boltzmann_abs_mean",
      "spectrum_conf0_boltzmann_abs_std",
      "spectrum_conf0_boltzmann_abs_skewness",
      "spectrum_conf0_boltzmann_abs_max_val",
      "spectrum_conf0_boltzmann_abs_p95",
      "spectrum_conf0_boltzmann_abs_peak_to_mean",
      "spectrum_conf0_boltzmann_pos_mean",
      "spectrum_conf0_boltzmann_pos_std",
      "spectrum_conf0_boltzmann_pos_skewness",
      "spectrum_conf0_boltzmann_shape_mean",
      "spectrum_conf0_boltzmann_shape_std",
      "spectrum_conf0_boltzmann_shape_skewness"
    ],
    "graphmol_boltzmann": [
      "spectrum_boltzmann_abs_mean",
      "spectrum_boltzmann_abs_std",
      "spectrum_boltzmann_abs_skewness",
      "spectrum_boltzmann_abs_max_val",
      "spectrum_boltzmann_abs_p95",
      "spectrum_boltzmann_abs_peak_to_mean",
      "spectrum_boltzmann_pos_mean",
      "spectrum_boltzmann_pos_std",
      "spectrum_boltzmann_pos_skewness",
      "spectrum_boltzmann_shape_mean",
      "spectrum_boltzmann_shape_std",
      "spectrum_boltzmann_shape_skewness"
    ],
    "graphmol_dipole": [  
      "dipole_Boltzmann_average",
      "spectrum_boltzmann_abs_mean",
      "spectrum_boltzmann_abs_std",
      "spectrum_boltzmann_abs_skewness",
      "spectrum_boltzmann_abs_max_val",
      "spectrum_boltzmann_abs_p95",
      "spectrum_boltzmann_abs_peak_to_mean",
      "spectrum_boltzmann_pos_mean",
      "spectrum_boltzmann_pos_std",
      "spectrum_boltzmann_pos_skewness",
      "spectrum_boltzmann_shape_mean",
      "spectrum_boltzmann_shape_std",
      "spectrum_boltzmann_shape_skewness"
    ]
    }

# Cross-validation and resampling config
N_SPLITS: int = 10
N_PERM: int = 500
N_BOOT: int = 1000


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class CVConfig:
    n_splits: int = N_SPLITS
    shuffle: bool = True
    random_state: int = 42


@dataclass
class RunConfig:
    data_path: str = DATA_PATH
    out_dir: str = OUT_DIR
    family_to_cols: Dict[str, List[str]] = None  # type: ignore[assignment]
    n_splits: int = N_SPLITS
    n_perm: int = N_PERM
    n_boot: int = N_BOOT

    def __post_init__(self) -> None:
        if self.family_to_cols is None:
            self.family_to_cols = FAMILY_TO_COLS


# ------------------------------
# Utility functions
# ------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from CSV or Parquet."""
    _, ext = os.path.splitext(path)
    if ext.lower() in {".csv"}:
        return pd.read_csv(path)
    if ext.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset extension: {ext}")


def _unique_non_nan_count(series: pd.Series) -> int:
    values = series.dropna().unique()
    return len(values)


def split_families(df: pd.DataFrame, mapping: Mapping[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Build per-family feature frames; drop all-NaN and constant columns."""
    out: Dict[str, pd.DataFrame] = {}
    for family, cols in mapping.items():
        cols_present = [c for c in cols if c in df.columns]
        Xf = df[cols_present].copy() if cols_present else pd.DataFrame(index=df.index)
        # Drop all-NaN columns
        Xf = Xf.dropna(axis=1, how="all")
        # Drop constant columns (after NaN removal)
        keep_cols: List[str] = []
        for c in Xf.columns:
            if _unique_non_nan_count(Xf[c]) > 1:
                keep_cols.append(c)
        Xf = Xf[keep_cols]
        out[family] = Xf
    return out


def bhattacharyya_distance(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Bhattacharyya distance between two Gaussian-fitted score sets."""
    eps = 1e-8
    mu1 = float(np.mean(pos_scores))
    mu2 = float(np.mean(neg_scores))
    var1 = float(np.var(pos_scores, ddof=1)) if pos_scores.size > 1 else 0.0
    var2 = float(np.var(neg_scores, ddof=1)) if neg_scores.size > 1 else 0.0
    var1 = max(var1, eps)
    var2 = max(var2, eps)
    term1 = 0.25 * math.log(0.25 * (var1 / var2 + var2 / var1 + 2.0))
    term2 = 0.25 * ((mu1 - mu2) ** 2) / (var1 + var2)
    return float(term1 + term2)


def bootstrap_ci(values: ArrayLike, n_boot: int, ci: float = 0.95) -> Tuple[float, float]:
    """Return (low, high) bootstrap percentile CI for 1D values."""
    rng = np.random.default_rng(42)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size == 0:
        return (np.nan, np.nan)
    n = values_arr.size
    alpha = (1.0 - ci) / 2.0
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values_arr[idx]
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return (lo, hi)


def compute_confusion_normalized(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """Row-normalized TN, FP, FN, TP for a binary confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    support_N = int(np.sum(y_true == 0))
    support_FN = int(np.sum(y_true == 1))
    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])
    tp = float(cm[1, 1])
    tn_norm = tn / support_N if support_N > 0 else np.nan
    fp_norm = fp / support_N if support_N > 0 else np.nan
    fn_norm = fn / support_FN if support_FN > 0 else np.nan
    tp_norm = tp / support_FN if support_FN > 0 else np.nan
    return tn_norm, fp_norm, fn_norm, tp_norm


# ------------------------------
# Modeling helpers
# ------------------------------

def _make_binary_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("vt", VarianceThreshold(threshold=1e-6)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",                    
                    C=3.0,
                    tol=1e-6,
                    max_iter=10000,
                    class_weight="balanced",
                    random_state=42
                ),
            ),
        ]
    )


def _make_multiclass_estimator() -> OneVsRestClassifier:
    base = _make_binary_pipeline()
    return OneVsRestClassifier(base, n_jobs=-1)


def _prepare_cv_splits(y: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return [(tr, te) for tr, te in skf.split(np.zeros_like(y), y)]


# ------------------------------
# Evaluation per family
# ------------------------------

def evaluate_binary_family(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_perm: int,
    n_boot: int,
) -> Dict[str, Any]:
    """Binary evaluation per family with CV, permutation test, and summaries."""
    n_splits = len(cv_splits)
    fold_metrics: List[Dict[str, float]] = []
    heldout_scores: List[float] = []
    heldout_true: List[int] = []
    heldout_ids: List[Any] = []
    heldout_fold: List[int] = []
    heldout_preds: List[int] = []

    for fold_id, (tr, te) in enumerate(tqdm(cv_splits, desc="Binary CV folds", leave=False), start=1):
        model = _make_binary_pipeline()
        model.fit(X[tr], y[tr])
        # Positive-class probabilities
        proba = model.predict_proba(X[te])[:, 1]
        pred_labels = (proba >= 0.5).astype(int)

        auprc = float(average_precision_score(y[te], proba))
        bal_acc = float(balanced_accuracy_score(y[te], pred_labels))
        mcc = float(matthews_corrcoef(y[te], pred_labels))
        auroc = float(roc_auc_score(y[te], proba))
        fold_metrics.append(
            {"AUPRC": auprc, "BalancedAccuracy": bal_acc, "MCC": mcc, "AUROC": auroc}
        )

        heldout_scores.extend(proba.tolist())
        heldout_true.extend(y[te].tolist())
        heldout_ids.extend(ids[te].tolist())
        heldout_fold.extend([fold_id] * te.size)
        heldout_preds.extend(pred_labels.tolist())

    # Aggregated observed metrics (on concatenated held-out predictions)
    y_concat = np.asarray(heldout_true, dtype=int)
    s_concat = np.asarray(heldout_scores, dtype=float)
    p_concat = (s_concat >= 0.5).astype(int)

    observed_auprc = float(average_precision_score(y_concat, s_concat))
    observed_bal_acc = float(balanced_accuracy_score(y_concat, p_concat))
    observed_mcc = float(matthews_corrcoef(y_concat, p_concat))
    observed_auroc = float(roc_auc_score(y_concat, s_concat))

    # Separability (Bhattacharyya distance) on held-out scores
    pos_scores = s_concat[y_concat == 1]
    neg_scores = s_concat[y_concat == 0]
    bhatt = float(bhattacharyya_distance(pos_scores, neg_scores))

    # Normalized confusion aggregated across held-out predictions
    tn_norm, fp_norm, fn_norm, tp_norm = compute_confusion_normalized(y_concat, p_concat)
    supp_N = int(np.sum(y_concat == 0))
    supp_FN = int(np.sum(y_concat == 1))

    # Chance AUPRC: prevalence of positive class
    prevalence_pos = float(np.mean(y))
    chance_auprc = prevalence_pos

    # Permutation test (AUPRC) with fixed CV splits
    rng = np.random.default_rng(42)
    perm_pr_values = np.empty(n_perm, dtype=float)
    perm_roc_values = np.empty(n_perm, dtype=float)

    for i in tqdm(range(n_perm), desc="Binary permutation test", leave=False):
        y_perm = y.copy()
        rng.shuffle(y_perm)

        heldout_perm_scores: List[float] = []
        heldout_perm_true: List[int] = []

        # reuse the SAME CV splits for fair comparison
        for (tr, te) in cv_splits:
            model = _make_binary_pipeline()
            model.fit(X[tr], y_perm[tr])
            proba = model.predict_proba(X[te])[:, 1]
            heldout_perm_scores.extend(proba.tolist())
            heldout_perm_true.extend(y_perm[te].tolist())

        y_perm_concat = np.asarray(heldout_perm_true, dtype=int)
        s_perm_concat = np.asarray(heldout_perm_scores, dtype=float)

        # Null metrics for this permutation
        if len(np.unique(y_perm_concat)) < 2:
            perm_roc_values[i] = 0.5
            perm_pr_values[i] = float(np.mean(y_perm_concat)) # Class prevalence
        else:
            perm_pr_values[i] = average_precision_score(y_perm_concat, s_perm_concat)
            perm_roc_values[i] = roc_auc_score(y_perm_concat, s_perm_concat)

    # One-sided p-values: P(null >= observed)
    p_value_auprc = float((1.0 + np.sum(perm_pr_values >= observed_auprc)) / (1.0 + n_perm))
    p_value_auroc = float((1.0 + np.sum(perm_roc_values >= observed_auroc)) / (1.0 + n_perm))

    # Bootstrap CIs over folds
    fold_auprc = np.array([m["AUPRC"] for m in fold_metrics], dtype=float)
    fold_balacc = np.array([m["BalancedAccuracy"] for m in fold_metrics], dtype=float)
    fold_mcc = np.array([m["MCC"] for m in fold_metrics], dtype=float)
    fold_auroc = np.array([m["AUROC"] for m in fold_metrics], dtype=float)
    ci_auprc = bootstrap_ci(fold_auprc, n_boot=n_boot)
    ci_balacc = bootstrap_ci(fold_balacc, n_boot=n_boot)
    ci_mcc = bootstrap_ci(fold_mcc, n_boot=n_boot)
    ci_auroc = bootstrap_ci(fold_auroc, n_boot=n_boot)
    # Assemble outputs
    scores_df = pd.DataFrame(
        {
            "molecule_id": heldout_ids,
            "family": None,  # to be filled by caller
            "y_true": heldout_true,
            "y_score": heldout_scores,
            "cv_fold": heldout_fold,
        }
    )

    perm_df = pd.DataFrame({
            "family": None, 
            "perm_id": np.arange(1, n_perm + 1), 
            "auprc_null": perm_pr_values, 
            "auroc_null": perm_roc_values   
        })

    summary = {
        "AUPRC": {
            "mean": float(fold_auprc.mean()),
            "std": float(fold_auprc.std(ddof=1)) if n_splits > 1 else 0.0,
            "ci_low": ci_auprc[0],
            "ci_high": ci_auprc[1],
            "observed": observed_auprc,
            "p_value": p_value_auprc,
        },
        "BalancedAccuracy": {
            "mean": float(fold_balacc.mean()),
            "std": float(fold_balacc.std(ddof=1)) if n_splits > 1 else 0.0,
            "ci_low": ci_balacc[0],
            "ci_high": ci_balacc[1],
            "observed": observed_bal_acc,
            "p_value": np.nan,
        },
        "MCC": {
            "mean": float(fold_mcc.mean()),
            "std": float(fold_mcc.std(ddof=1)) if n_splits > 1 else 0.0,
            "ci_low": ci_mcc[0],
            "ci_high": ci_mcc[1],
            "observed": observed_mcc,
            "p_value": np.nan,
        },
        "BhattacharyyaD": {
            "mean": bhatt,
            "std": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "observed": bhatt,
            "p_value": np.nan,
        },
        "AUROC": {
            "mean": float(fold_auroc.mean()),
            "std": float(fold_auroc.std(ddof=1)) if n_splits > 1 else 0.0,
            "ci_low": ci_auroc[0],
            "ci_high": ci_auroc[1],
            "observed": observed_auroc,
            "p_value": p_value_auroc,
        },
        "n_folds": n_splits,
        "n_perm": n_perm,
        "prevalence_positive": prevalence_pos,
        "chance_auprc": chance_auprc,
        "confusion_norm": {
            "tn": tn_norm,
            "fp": fp_norm,
            "fn": fn_norm,
            "tp": tp_norm,
            "support_N": supp_N,
            "support_FN": supp_FN,
        },
    }

    return {
        "fold_metrics": fold_metrics,
        "scores_df": scores_df,
        "perm_df": perm_df,
        "summary": summary,
    }


def evaluate_multiclass_family(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_perm: int,
    n_boot: int,
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """Multiclass evaluation (OVR) with CV, permutation test, and summaries."""
    classes = np.array(sorted(np.unique(y)))
    n_classes = classes.size
    class_to_index = {c: i for i, c in enumerate(classes)}

    fold_macro_auprc: List[float] = []
    fold_macro_balacc: List[float] = []

    # For observed metrics computed on concatenated holds
    all_true: List[int] = []
    all_proba: List[np.ndarray] = []
    all_pred: List[int] = []
    mc_score_rows: List[Dict[str, Any]] = []

    for (tr, te) in tqdm(cv_splits, desc="Multiclass CV folds", leave=False):
        est = _make_multiclass_estimator()
        est.fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])  # shape (n_te, n_classes)
        pred = np.asarray([classes[np.argmax(row)] for row in proba], dtype=int)

        for j, mol_id in enumerate(ids[te]):
            true_lbl = int(y[te][j])
            for ci, c in enumerate(classes):
                score = float(proba[j, ci])
                mc_score_rows.append({
                                "molecule_id": mol_id,
                                "family": family if family is not None else "",
                                "cv_fold": int(len(fold_macro_auprc) + 1),  # 1-based fold index
                                "true_lbl": true_lbl,
                                "class_lbl": int(c),
                                "score": score,
                })   

        # Macro-AUPRC (OVR): mean over classes of AP for one-vs-rest
        fold_aps: List[float] = []
        for ci, c in enumerate(classes):
            y_true_bin = (y[te] == c).astype(int)            
            if y_true_bin.sum() == 0:
                continue
            fold_aps.append(float(average_precision_score(y_true_bin, proba[:, ci])))
        macro_auprc = float(np.mean(fold_aps)) if fold_aps else float("nan")

        macro_balacc = float(balanced_accuracy_score(y[te], pred))
        fold_macro_auprc.append(macro_auprc)
        fold_macro_balacc.append(macro_balacc)

        all_true.extend(y[te].tolist())
        all_proba.append(proba)
        all_pred.extend(pred.tolist())

    # Observed on concatenated
    y_concat = np.asarray(all_true, dtype=int)
    proba_concat = np.vstack(all_proba) if all_proba else np.empty((0, n_classes))
    pred_concat = np.asarray(all_pred, dtype=int)

    obs_aps: List[float] = []
    for ci, c in enumerate(classes):
        y_true_bin = (y_concat == c).astype(int)
        if y_true_bin.sum() == 0:
            continue
        obs_aps.append(float(average_precision_score(y_true_bin, proba_concat[:, ci])))

    observed_macro_auprc = float(np.mean(obs_aps)) if obs_aps else float("nan")

    observed_macro_balacc = float(balanced_accuracy_score(y_concat, pred_concat))

    # Chance macro-AUPRC: mean of per-class prevalences
    prevalences = [(y == c).mean() for c in classes]
    chance_macro_auprc = float(np.mean(prevalences))

    # Permutation (Macro-AUPRC) with fixed splits
    rng = np.random.default_rng(42)
    perm_values = np.empty(n_perm, dtype=float)
    for i in tqdm(range(n_perm), desc="Multiclass permutation test", leave=False):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        # Accumulate predictions across splits
        perm_true_all: List[int] = []
        perm_proba_all: List[np.ndarray] = []

        for (tr, te) in cv_splits:
            est = _make_multiclass_estimator()
            est.fit(X[tr], y_perm[tr])
            proba = est.predict_proba(X[te])
            perm_proba_all.append(proba)
            perm_true_all.extend(y_perm[te].tolist())
      
        perm_true_arr = np.asarray(perm_true_all, dtype=int)
        perm_proba_arr = np.vstack(perm_proba_all)
        
        aps: List[float] = []
        for ci, c in enumerate(classes):
            y_true_bin = (perm_true_arr == c).astype(int)
            
            if y_true_bin.sum() == 0:
                continue
            aps.append(float(average_precision_score(y_true_bin, perm_proba_arr[:, ci])))
        perm_values[i] = float(np.mean(aps)) if aps else float("nan")

    valid_mask = np.isfinite(perm_values)
    if n_perm > 0 and np.isfinite(observed_macro_auprc) and valid_mask.any():
        perm_vals = perm_values[valid_mask]
        p_value = float((1.0 + float(np.sum(perm_vals >= observed_macro_auprc))) / (1.0 + perm_vals.size))
    else:
        p_value = float("nan")

    # Bootstrap CIs over folds
    ci_macro_auprc = bootstrap_ci(np.asarray(fold_macro_auprc), n_boot=n_boot)
    ci_macro_balacc = bootstrap_ci(np.asarray(fold_macro_balacc), n_boot=n_boot)

    perm_df = pd.DataFrame(
        {
            "family": family,
            "perm_id": np.arange(1, n_perm + 1),
            "macro_auprc_null": perm_values,
        }
    )

    summary = {
        "MacroAUPRC": {
            "mean": float(np.mean(fold_macro_auprc)),
            "std": float(np.std(fold_macro_auprc, ddof=1)) if len(fold_macro_auprc) > 1 else 0.0,
            "ci_low": ci_macro_auprc[0],
            "ci_high": ci_macro_auprc[1],
            "observed": observed_macro_auprc,
            "p_value": p_value,
        },
        "MacroBalancedAccuracy": {
            "mean": float(np.mean(fold_macro_balacc)),
            "std": float(np.std(fold_macro_balacc, ddof=1)) if len(fold_macro_balacc) > 1 else 0.0,
            "ci_low": ci_macro_balacc[0],
            "ci_high": ci_macro_balacc[1],
            "observed": observed_macro_balacc,
            "p_value": np.nan,
        },
        "n_folds": len(cv_splits),
        "n_perm": n_perm,
        "chance_macro_auprc": chance_macro_auprc,
    }

    # Fold metrics dataframe-style for saving
    fold_metrics = [
        {"MacroAUPRC": float(a), "MacroBalancedAccuracy": float(b)}
        for a, b in zip(fold_macro_auprc, fold_macro_balacc)
    ]

    scores_df = pd.DataFrame(mc_score_rows, columns = [
        "molecule_id", "family", "cv_fold", "true_lbl", "class_lbl", "score"
    ])

    return {
        "fold_metrics": fold_metrics,
        "perm_df": perm_df,
        "summary": summary,
        "scores": scores_df,
    }


# ------------------------------
# Saving outputs
# ------------------------------

def save_all_outputs(
    out_dir: str,
    binary_results: Dict[str, Dict[str, Any]],
    multiclass_results: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    """Write CSV result files and meta JSON for binary/multiclass outputs."""
    os.makedirs(out_dir, exist_ok=True)

    # Binary summaries
    bin_summary_rows: List[Dict[str, Any]] = []
    bin_fold_rows: List[Dict[str, Any]] = []
    bin_confusion_rows: List[Dict[str, Any]] = []
    bin_scores_rows: List[pd.DataFrame] = []
    bin_perm_rows: List[pd.DataFrame] = []

    for family, res in binary_results.items():
        summ = res["summary"]
        # Summary rows across metrics
        for metric_name in ["AUPRC", "BalancedAccuracy", "MCC", "AUROC", "BhattacharyyaD"]:
            m = summ[metric_name]
            bin_summary_rows.append(
                {
                    "family": family,
                    "metric": metric_name,
                    "mean": m.get("mean", np.nan),
                    "std": m.get("std", np.nan),
                    "ci_low": m.get("ci_low", np.nan),
                    "ci_high": m.get("ci_high", np.nan),
                    "observed": m.get("observed", np.nan),"p_value": m.get("p_value", np.nan) if metric_name in ("AUPRC", "AUROC") else np.nan,
                    "n_folds": summ.get("n_folds", np.nan),
                    "n_perm": summ.get("n_perm", np.nan),
                    "prevalence_positive": summ.get("prevalence_positive", np.nan),
                    "chance_auprc": summ.get("chance_auprc", np.nan),
                }
            )

        # Fold metrics
        for i, fm in enumerate(res["fold_metrics"], start=1):
            bin_fold_rows.append(
                {
                    "family": family,
                    "fold": i,
                    "AUPRC": fm["AUPRC"],
                    "BalancedAccuracy": fm["BalancedAccuracy"],
                    "MCC": fm["MCC"],
                    "AUROC": fm["AUROC"],
                }
            )

        # Confusion normalized
        cn = summ.get("confusion_norm", {})
        bin_confusion_rows.append(
            {
                "family": family,
                "tn": cn.get("tn", np.nan),
                "fp": cn.get("fp", np.nan),
                "fn": cn.get("fn", np.nan),
                "tp": cn.get("tp", np.nan),
                "support_N": cn.get("support_N", np.nan),
                "support_FN": cn.get("support_FN", np.nan),
            }
        )

        # Scores and permutations
        sdf: pd.DataFrame = res["scores_df"].copy()
        sdf["family"] = family
        bin_scores_rows.append(sdf)

        pdf: pd.DataFrame = res["perm_df"].copy()
        pdf["family"] = family
        bin_perm_rows.append(pdf)

    # Multiclass summaries
    mc_summary_rows: List[Dict[str, Any]] = []
    mc_fold_rows: List[Dict[str, Any]] = []
    mc_perm_rows: List[pd.DataFrame] = []
    mc_scores_all: List[pd.DataFrame] = []

    for family, res in multiclass_results.items():
        summ = res["summary"]
        for metric_name in ["MacroAUPRC", "MacroBalancedAccuracy"]:
            m = summ[metric_name]
            mc_summary_rows.append(
                {
                    "family": family,
                    "metric": metric_name,
                    "mean": m.get("mean", np.nan),
                    "std": m.get("std", np.nan),
                    "ci_low": m.get("ci_low", np.nan),
                    "ci_high": m.get("ci_high", np.nan),
                    "observed": m.get("observed", np.nan),
                    "p_value": m.get("p_value", np.nan) if metric_name == "MacroAUPRC" else np.nan,
                    "n_folds": summ.get("n_folds", np.nan),
                    "n_perm": summ.get("n_perm", np.nan),
                    "chance_macro_auprc": summ.get("chance_macro_auprc", np.nan),
                }
            )

        for i, fm in enumerate(res["fold_metrics"], start=1):
            mc_fold_rows.append(
                {
                    "family": family,
                    "fold": i,
                    "MacroAUPRC": fm["MacroAUPRC"],
                    "MacroBalancedAccuracy": fm["MacroBalancedAccuracy"],
                }
            )

        pdf: pd.DataFrame = res["perm_df"].copy()
        pdf["family"] = family
        mc_perm_rows.append(pdf)

        scores: pd.DataFrame = res["scores"].copy()
        mc_scores_all.append(scores)    

    # Write files
    pd.DataFrame(bin_summary_rows).to_csv(
        os.path.join(out_dir, "cLR_binary_summary.csv"), index=False
    )
    pd.DataFrame(bin_fold_rows).to_csv(
        os.path.join(out_dir, "cLR_binary_fold_metrics.csv"), index=False
    )
    pd.DataFrame(bin_confusion_rows).to_csv(
        os.path.join(out_dir, "cLR_binary_confusion_normalized.csv"), index=False
    )
    (pd.concat(bin_scores_rows, ignore_index=True) if bin_scores_rows else pd.DataFrame(columns=["molecule_id","family","y_true","y_score","cv_fold"]))\
        .to_csv(os.path.join(out_dir, "cLR_binary_scores.csv"), index=False)
    (pd.concat(bin_perm_rows, ignore_index=True) if bin_perm_rows else pd.DataFrame(columns=["family","perm_id","auprc_null","auroc_null"]))\
        .to_csv(os.path.join(out_dir, "cLR_binary_permutation_nulls.csv"), index=False)

    pd.DataFrame(mc_summary_rows).to_csv(
        os.path.join(out_dir, "cLR_multiclass_summary.csv"), index=False
    )
    pd.DataFrame(mc_fold_rows).to_csv(
        os.path.join(out_dir, "cLR_multiclass_fold_metrics.csv"), index=False
    )
    (pd.concat(mc_perm_rows, ignore_index=True) if mc_perm_rows else pd.DataFrame(columns=["family","perm_id","macro_auprc_null"]))\
        .to_csv(os.path.join(out_dir, "cLR_multiclass_permutation_macroauprc.csv"), index=False)

    (pd.concat(mc_scores_all, ignore_index=True) if mc_scores_all else pd.DataFrame(
        columns=["molecule_id", "family", "cv_fold", "true_lbl", "class_lbl", "score"])
    ).to_csv(os.path.join(out_dir, "cLR_multiclass_scores.csv"), index=False)

    # Meta JSON
    meta_path = os.path.join(out_dir, "cLR_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ------------------------------
# Main execution
# ------------------------------

def _collect_meta(
    cfg: RunConfig,
    df: pd.DataFrame,
    families_clean: Dict[str, pd.DataFrame],
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    dropped_by_family: Dict[str, List[str]],
) -> Dict[str, Any]:
    import sklearn  # local import for version capture

    cls_counts = df["phase"].value_counts(dropna=False).to_dict() if "phase" in df.columns else {}
    trans_counts = df["transition_type"].value_counts(dropna=False).to_dict() if "transition_type" in df.columns else {}
    
    # Get pipeline information
    binary_pipeline = _make_binary_pipeline()
    multiclass_estimator = _make_multiclass_estimator()
    
    # Extract pipeline steps and parameters
    pipeline_steps = []
    for step_name, step_estimator in binary_pipeline.steps:
        step_info = {
            "name": step_name,
            "class": step_estimator.__class__.__name__,
            "module": step_estimator.__class__.__module__
        }
        
        # Add specific parameters for each step
        if hasattr(step_estimator, 'get_params'):
            params = step_estimator.get_params()
            # Filter out non-serializable parameters and keep only relevant ones
            serializable_params = {}
            for key, value in params.items():
                if key.startswith('_'):
                    continue  # Skip private attributes
                try:
                    # Test if the value is JSON serializable
                    json.dumps(value)
                    serializable_params[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable values to strings
                    serializable_params[key] = str(value)
            step_info["parameters"] = serializable_params
        
        pipeline_steps.append(step_info)
    
    # Get multiclass estimator info
    multiclass_info = {
        "class": multiclass_estimator.__class__.__name__,
        "module": multiclass_estimator.__class__.__module__,
        "n_jobs": getattr(multiclass_estimator, 'n_jobs', None)
    }
    
    meta: Dict[str, Any] = {
        "dataset_path": cfg.data_path,
        "timestamp": datetime.now().isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_samples": int(df.shape[0]),
        "class_counts_phase": cls_counts,
        "class_counts_transition_type": trans_counts,
        "FAMILY_TO_COLS": cfg.family_to_cols,
        "cv_config": {"n_splits": cfg.n_splits, "shuffle": True, "random_state": 42},
        "n_perm": cfg.n_perm,
        "n_boot": cfg.n_boot,
        "families_kept_n_features": {k: int(v.shape[1]) for k, v in families_clean.items()},
        "families_dropped_columns": dropped_by_family,
        "pipeline": {
            "binary_pipeline": {
                "steps": pipeline_steps,
                "description": "Logistic Regression with L2 penalty, balanced class weights"
            },
            "multiclass_estimator": multiclass_info,
            "description": "One-vs-Rest classifier using binary pipeline for multiclass classification"
        }
    }
    return meta


# no CLI integration: call main() from notebooks or scripts if desired
