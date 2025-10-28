"""Evaluate descriptor families (tree models) and write CSV/JSON results."""

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
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
# import shap  # Disabled - using permutation_importance instead

from joblib import Memory, dump, load
import tempfile
import hashlib
from tqdm.auto import tqdm

# Global reproducibility
np.random.seed(42)

# Create a memory cache for preprocessing steps
CACHE_DIR = tempfile.mkdtemp(prefix="classifierRF_cache_")
memory = Memory(location=CACHE_DIR, verbose=0)


# ------------------------------
# Custom XGBoost wrapper with early stopping
# ------------------------------

class XGBClassifierWithEarlyStopping(BaseEstimator, ClassifierMixin):
    """XGBoost classifier with built-in early stopping for use in sklearn pipelines.
    
    This wrapper handles the creation of validation sets internally and applies
    early stopping properly within sklearn pipelines by ensuring the validation
    data goes through the same preprocessing steps as the training data.
    """
    
    def _more_tags(self):
        return {
            'binary_only': False,
            'requires_fit': True,
            'requires_y': True,
            'requires_positive_X': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            '_xfail_checks': {},
        }
    
    def __init__(self, 
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=3,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 min_child_weight=5,
                 reg_lambda=1.0,
                 tree_method="hist",
                 objective="binary:logistic",
                 eval_metric="aucpr",
                 scale_pos_weight=1.0,
                 early_stopping_rounds=50,
                 validation_fraction=0.15,
                 n_jobs=-1,
                 random_state=42,
                 **kwargs):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.objective = objective
        self.eval_metric = eval_metric
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Store any additional kwargs
        self.kwargs = kwargs
        
        # Initialize the underlying XGBoost model
        self.model_ = None
        
    def fit(self, X, y):
        """Fit the XGBoost model with early stopping."""
        # Split training data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.validation_fraction, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Create the XGBoost model with current parameters including early stopping
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            tree_method=self.tree_method,
            objective=self.objective,
            eval_metric=self.eval_metric,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,  # Set in constructor
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs
        )
        
        # Fit with evaluation set (early stopping configured in constructor)
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Set classes_ attribute for sklearn compatibility
        self.classes_ = self.model_.classes_
        self.n_classes_ = len(self.classes_)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model_.predict_proba(X)
    
    @property
    def feature_importances_(self):
        """Get feature importances from the fitted model."""
        if self.model_ is None:
            raise ValueError("Model must be fitted before accessing feature importances")
        return self.model_.feature_importances_
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'reg_lambda': self.reg_lambda,
            'tree_method': self.tree_method,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'scale_pos_weight': self.scale_pos_weight,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_fraction': self.validation_fraction,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


# ------------------------------
# Custom RF wrapper with early stopping
# ------------------------------
class RFWithOOBEarlyStop(RandomForestClassifier):
    """
    Grow a RandomForest in batches using warm_start and stop when the OOB AUPRC
    stops improving by >= tol for `patience` consecutive batches.

    Note: OOB decision_function is not exposed directly; we reconstruct
    positive-class probs from OOB votes via oob_decision_function_.
    """
    def __init__(self,
                 start_estimators=200,
                 step_estimators=200,
                 max_estimators=1600,
                 patience=2,
                 tol=1e-3,
                 verbose=1,
                 **kwargs):
        super().__init__(n_estimators=start_estimators, warm_start=True, oob_score=True, **kwargs)
        self.start_estimators = start_estimators
        self.step_estimators = step_estimators
        self.max_estimators = max_estimators
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.oob_history_ = []
        self.best_n_estimators_ = None
        self.best_score_ = None

    def _oob_auprc(self, y):
        # Works for binary classification: positive class is column 1
        if self.oob_decision_function_ is None:
            return np.nan
        y_true = np.asarray(y, dtype=int)
        s = self.oob_decision_function_[:, 1]
        # Handle case where all predictions are the same (can cause issues with AUPRC)
        if len(np.unique(s)) <= 1:
            return np.nan
        return float(average_precision_score(y_true, s))

    def fit(self, X, y, sample_weight=None):
        no_improve = 0
        best = -np.inf
        best_n = self.n_estimators

        while True:
            super().fit(X, y, sample_weight=sample_weight)
            if not hasattr(self, "oob_decision_function_") or self.oob_decision_function_ is None:
                # OOB requires bootstrap=True and enough trees
                score = np.nan
            else:
                score = self._oob_auprc(y)
            self.oob_history_.append((self.n_estimators, score))

            if self.verbose:
                print(f"[RF-OOB] n_estimators={self.n_estimators}, OOB AUPRC={score:.4f}")

            # Only update best if we have a valid score
            if not np.isnan(score) and score > best + self.tol:
                best = score
                best_n = self.n_estimators
                no_improve = 0
            elif not np.isnan(score):  # Valid score but not improving
                no_improve += 1
            # If score is NaN, don't increment no_improve (give it more chances)

            if no_improve >= self.patience or self.n_estimators + self.step_estimators > self.max_estimators:
                break

            # grow more trees
            self.n_estimators += self.step_estimators

        # Refit at best_n (optional; forest already contains all trees)
        self.best_n_estimators_ = best_n
        self.best_score_ = best
        if self.verbose:
            print(f"[RF-OOB] Early stop at {self.n_estimators} (best @ {best_n}, AUPRC={best:.4f})")
        return self
# ------------------------------
# Configuration (editable)
# ------------------------------

# Default dataset path: try repository-level merged descriptors if present
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

DATA_PATH: str = os.path.abspath(
    os.path.join(_REPO_ROOT, "data", "full_dataset.csv")
)

# Output directory: output_ml/classifier_RF/results
OUT_DIR: str = os.path.abspath(
    os.path.join(_REPO_ROOT, "data", "ml", "classifier_RF", "results")
)

# Model saving directory: output_ml/classifier_RF/models
MODEL_DIR: str = os.path.abspath(
    os.path.join(_REPO_ROOT, "data", "ml", "classifier_RF", "models")
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
N_SPLITS: int = 5
N_PERM: int = 20
N_BOOT: int = 200
MODEL_TYPE: str = "rf"

# Family-specific variance thresholds
FAMILY_VARIANCE_THRESHOLDS: Dict[str, float] = {
    "dipole_prop": 0.1,          # Single continuous property - very low threshold
    "sigma_profile": 1e-6,        # Statistical moments and magnitudes  
    "graphmol_boltzmann": 1e-6,   # Boltzmann-weighted spectrum features 
    "graphmol_conf0": 1e-6,       # Single conformer spectrum features 
    "graphmol_dipole": 1e-6,       # Single conformer spectrum features 
}

FAMILY_ESTIMATORS: Dict[str, int] = {
    "dipole_prop": 200,
    "sigma_profile": 1000,
    "graphmol_boltzmann": 1000,
    "graphmol_conf0": 1000,
    "graphmol_dipole": 1000
}
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
    model_dir: str = MODEL_DIR
    family_to_cols: Dict[str, List[str]] = None  # type: ignore[assignment]
    n_splits: int = N_SPLITS
    n_perm: int = N_PERM
    n_boot: int = N_BOOT
    model_type: str = MODEL_TYPE
    save_models: bool = True

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



def bootstrap_ci(values: ArrayLike, n_boot: int, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap percentile confidence interval over fold-level metric values.

    Args:
        values: 1D array-like of observed fold-level metrics.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95).

    Returns:
        (ci_low, ci_high) percentile bounds.
    """
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
    """Compute normalized confusion matrix entries for binary case.

    Normalization is by true labels (row-wise). Returns entries corresponding to
    [[tn, fp], [fn, tp]] normalized by support of each true class.

    Args:
        y_true: True binary labels {0,1}.
        y_pred: Predicted binary labels {0,1}.

    Returns:
        (tn_norm, fp_norm, fn_norm, tp_norm)
    """
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



def _clf_feature_names(model, orig_names):
    vt = model.named_steps["vt"]
    if vt is None:
        return list(orig_names)
    mask = vt.get_support()
    return [n for n, keep in zip(orig_names, mask) if keep]

@memory.cache
def _fit_preprocessing_steps(X_train, y_train, cfg=None):
    """Cache-fitting of imputer/variance-filter for reuse across splits."""
    # Create hash of training data for cache key
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    
    # Fit preprocessing steps
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_imputed = imputer.transform(X_train)
    
    vt = VarianceThreshold(threshold=1e-6).fit(X_imputed)
    X_vt = vt.transform(X_imputed)
    
    # No scaler in current pipeline, but ready for future use
    scaler = None
    
    return imputer, vt, scaler, X_vt

def _transform_to_clf_space(model, X):
    # Ensure input is numpy array
    Xt = np.asarray(X, dtype=float)
    if "imputer" in model.named_steps:
        Xt = model.named_steps["imputer"].transform(Xt)
    if "vt" in model.named_steps:
        Xt = model.named_steps["vt"].transform(Xt)
    if "scaler" in model.named_steps:
        Xt = model.named_steps["scaler"].transform(Xt)
    # Ensure output is numpy array
    return np.asarray(Xt, dtype=float)


def save_model(model, model_dir: str, family: str, task_type: str, fold: Optional[int] = None) -> str:
    """Serialize a model (or model+meta dict) to disk and return path."""
    os.makedirs(model_dir, exist_ok=True)
    
    if fold is not None:
        filename = f"{family}_{task_type}_fold{fold}.joblib"
        
    else:
        filename = f"{family}_{task_type}_final.joblib"
    
    model_path = os.path.join(model_dir, filename)
    dump(model, model_path)
    return model_path


def load_model(model_path: str):
    """Load a previously saved model object from disk."""
    return load(model_path)


def save_final_models(
    X: np.ndarray, 
    y: np.ndarray, 
    model_dir: str, 
    family: str, 
    task_type: str,
    feature_names: Optional[List[str]] = None,
    cfg: Optional[RunConfig] = None
) -> str:
    """Fit on all data and persist a final model for inference."""
    if task_type == "binary":
        model = _make_binary_pipeline(cfg=cfg, family=family)
        
        # For XGB, handle class imbalance
        if isinstance(model.named_steps.get("clf"), (XGBClassifier, XGBClassifierWithEarlyStopping)):
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            model.set_params(clf__scale_pos_weight=spw)
            
    elif task_type == "multiclass":
        model = _make_multiclass_estimator(cfg=cfg, family=family)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Train on full dataset
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    
    # Save model with metadata
    model_info = {
        'model': model,
        'family': family,
        'task_type': task_type,
        'feature_names': feature_names,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'timestamp': datetime.now().isoformat()
    }
    
    model_path = save_model(model_info, model_dir, family, task_type)
    return model_path


def predict_with_saved_model(model_path: str, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Load a saved model and return predictions and probabilities."""
    model_info = load_model(model_path)
    model = model_info['model']
    task_type = model_info['task_type']
    
    X_array = np.asarray(X, dtype=float)
    
    if task_type == "binary":
        probabilities = model.predict_proba(X_array)
        predictions = (probabilities[:, 1] >= 0.5).astype(int)
        return {
            'predictions': predictions,
            'probabilities': probabilities[:, 1],  # Positive class probability
            'all_probabilities': probabilities
        }
    elif task_type == "multiclass":
        probabilities = model.predict_proba(X_array)
        predictions = model.predict(X_array)
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'predicted_classes': predictions
        }
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
# ------------------------------
# Modeling helpers
# ------------------------------

def _make_binary_pipeline(cfg=None, family=None, class_weight=None) -> Pipeline:
    """Binary pipeline factory (RF or XGB) with family-specific options."""
    model_type = getattr(cfg, "model_type", "rf") if cfg is not None else "rf"
    if model_type.lower() == "xgb":
        return _make_binary_pipeline_XGB(family=family)
    else:
        return _make_binary_pipeline_RF(family=family, class_weight=class_weight)

def _make_binary_pipeline_RF(family=None, class_weight=None) -> Pipeline:
    # Get family-specific variance threshold
    var_threshold = FAMILY_VARIANCE_THRESHOLDS.get(family, 1e-6) if family else 1e-6
    
    # Use provided class_weight or default to "balanced_subsample"
    if class_weight is None:
        class_weight = "balanced_subsample"
    
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("vt", VarianceThreshold(threshold=var_threshold)),
            ("clf",RFWithOOBEarlyStop(
                    start_estimators=300,   # start
                    step_estimators=200,    # grow by 200
                    max_estimators=1600,    # cap
                    patience=2,             # stop after 2 non-improving steps
                    tol=1e-3,
                    verbose=0,
                    criterion="log_loss",
                    max_depth=None,
                    min_samples_leaf=1, # Try 1 next
                    min_samples_split=4,
                    max_features=0.3,       # try 0.3 next, 
                    bootstrap=True,
                    max_samples=0.6,        # subsample for speed
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1
                ),
            ),
        ]
    )

def _make_binary_pipeline_XGB(family=None) -> Pipeline:
    # Get family-specific variance threshold and estimator count
    estim = FAMILY_ESTIMATORS.get(family, 200) if family else 200
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifierWithEarlyStopping(
                    n_estimators=estim,          # Use family-specific estimator count
                    learning_rate=0.05,
                    max_depth=4,                 # try 3–5
                    subsample=0.8,
                    colsample_bytree=0.6,
                    min_child_weight=3,          # regularization via leaf min Hessian
                    reg_lambda=2.0,              # L2
                    reg_alpha=0.0,
                    gamma=0.0,
                    tree_method="hist",
                    objective="binary:logistic",
                    eval_metric="aucpr",         # or ["aucpr","auc"]
                    scale_pos_weight=1.0,        # set to (neg/pos) if classes are imbalanced
                    early_stopping_rounds=50,   # Enable early stopping
                    validation_fraction=0.2,   # Use 15% of training data for validation
                    n_jobs=-1,
                    random_state=42
            )),
        ]
    )

def _make_multiclass_estimator(cfg=None, family=None, class_weight=None) -> OneVsRestClassifier:
    base = _make_binary_pipeline(cfg=cfg, family=family, class_weight=class_weight)
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
    feature_names: Optional[List[str]] = None,
    family: Optional[str] = None,
    model_dir: Optional[str] = None,
    save_models: bool = False,
    cfg: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """Evaluate binary classification (N vs FN) for a descriptor family.

    Uses predetermined CV splits. Threshold of 0.5 is used for confusion matrix.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary labels (0 for N, 1 for FN).
        ids: Molecule identifiers.
        cv_splits: Precomputed list of (train_idx, test_idx).
        n_perm: Number of permutations for AUPRC null distribution.
        n_boot: Number of bootstrap resamples for CIs over folds.
        feature_names: Optional list of feature names.
        family: Optional descriptor family name.
        model_dir: Optional directory to save trained models.
        save_models: Whether to save trained models to disk.
        cfg: Optional RunConfig with model configuration.

    Returns:
        Dict with keys: fold_metrics, scores_df, confusion_norm, summary,
        perm_df, pos_prevalence, chance_auprc, saved_models.
    """
    n_splits = len(cv_splits)
    fold_metrics: List[Dict[str, float]] = []
    heldout_scores: List[float] = []
    heldout_true: List[int] = []
    heldout_ids: List[Any] = []
    heldout_fold: List[int] = []
    heldout_preds: List[int] = []
    
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    saved_model_paths: List[str] = []
    
    # Compute class weights once on the full dataset to avoid warm_start warnings
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

    for fold_id, (tr, te) in tqdm(enumerate(cv_splits, start=1), total=len(cv_splits),
                                    desc=f"{family} binary CV"):
        
        # Get training labels for this fold
        y_tr = y[tr]
        
        # Use precomputed class weights from full dataset to avoid warm_start warnings
        model = _make_binary_pipeline(cfg=cfg, family=family, class_weight=class_weight_dict)

        # Handle XGB class imbalance for both wrapper and regular XGBoost
        if isinstance(model.named_steps.get("clf"), (XGBClassifier, XGBClassifierWithEarlyStopping)):
            # per-fold imbalance (neg/pos) for scale_pos_weight
            n_pos = int((y_tr == 1).sum())
            n_neg = int((y_tr == 0).sum())
            spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
            model.set_params(clf__scale_pos_weight=spw)

        model.fit(np.asarray(X[tr], dtype=float), np.asarray(y[tr], dtype=int))
        
        # Save fold model if requested
        if save_models and model_dir and family:
            model_path = save_model(model, model_dir, family, "binary", fold=fold_id)
            saved_model_paths.append(model_path)
        
        # Positive-class probabilities
        proba = model.predict_proba(np.asarray(X[te], dtype=float))[:, 1]
        pred_labels = (proba >= 0.5).astype(int)

        auprc = float(average_precision_score(y[te], proba))
        auroc = float(roc_auc_score(y[te], proba))
        fold_metrics.append(
            {"AUPRC": auprc, "AUROC": auroc}
        )

        heldout_scores.extend(proba.tolist())
        heldout_true.extend(y[te].tolist())
        heldout_ids.extend(ids[te].tolist())
        heldout_fold.extend([fold_id] * te.size)
        heldout_preds.extend(pred_labels.tolist())

    # Aggregated observed metrics (on concatenated held-out predictions)
    y_concat = np.asarray(heldout_true, dtype=int)
    s_concat = np.asarray(heldout_scores, dtype=float)
    p_concat = np.asarray(heldout_preds, dtype=int)

    observed_auprc = float(average_precision_score(y_concat, s_concat))
    observed_auroc = float(roc_auc_score(y_concat, s_concat))

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

    for i in tqdm(range(n_perm), desc=f"{family} binary permutation"):
        y_perm = y.copy()
        rng.shuffle(y_perm)

        heldout_perm_scores: List[float] = []
        heldout_perm_true: List[int] = []

        # reuse the SAME CV splits for fair comparison
        for (tr, te) in cv_splits:
            # Use precomputed class weights from full dataset to avoid warm_start warnings
            model = _make_binary_pipeline(cfg=cfg, family=family, class_weight=class_weight_dict)
            model.fit(np.asarray(X[tr], dtype=float), np.asarray(y_perm[tr], dtype=int))
            proba = model.predict_proba(np.asarray(X[te], dtype=float))[:, 1]
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
    fold_auroc = np.array([m["AUROC"] for m in fold_metrics], dtype=float)
    ci_auprc = bootstrap_ci(fold_auprc, n_boot=n_boot)
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

    # Train and save final model on full dataset if requested
    final_model_path = None
    if save_models and model_dir and family:
        final_model_path = save_final_models(
            X=X, y=y, model_dir=model_dir, family=family, 
            task_type="binary", feature_names=feature_names, cfg=cfg
        )
    
    return {
        "fold_metrics": fold_metrics,
        "scores_df": scores_df,
        "perm_df": perm_df,
        "summary": summary,
        "saved_models": {
            "fold_models": saved_model_paths,
            "final_model": final_model_path
        }
    }


def evaluate_multiclass_family(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_perm: int,
    n_boot: int,
    family: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    model_dir: Optional[str] = None,
    save_models: bool = False,
    cfg: Optional[RunConfig] = None,
) -> Dict[str, Any]:
    """Evaluate multiclass classification (transition types 2/3/4/5).

    Args:
        X: Feature matrix.
        y: Integer class labels in {2,3,4,5} (will be treated as categorical).
        ids: Molecule identifiers (unused for outputs here; kept for symmetry).
        cv_splits: Precomputed CV splits.
        n_perm: Number of permutations for Macro-AUPRC null distribution.
        n_boot: Number of bootstrap resamples for CIs over folds.
        family: Optional descriptor family name.
        feature_names: Optional list of feature names.
        model_dir: Optional directory to save trained models.
        save_models: Whether to save trained models to disk.
        cfg: Optional RunConfig with model configuration.

    Returns:
        Dict with keys: fold_metrics, perm_df, summary, saved_models.
    """
    classes = np.array(sorted(np.unique(y)))
    n_classes = classes.size
    class_to_index = {c: i for i, c in enumerate(classes)}

    fold_macro_auprc: List[float] = []

    # For observed metrics computed on concatenated holds
    all_true: List[int] = []
    all_proba: List[np.ndarray] = []
    all_pred: List[int] = []
    mc_score_rows: List[Dict[str, Any]] = []
    
    saved_model_paths: List[str] = []
    
    # Compute class weights once on the full dataset to avoid warm_start warnings
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}

    for fold_idx, (tr, te) in tqdm(enumerate(cv_splits, start=1), total=len(cv_splits),
                                    desc=f"{family} multiclass CV"):
        # Get training labels for this fold
        y_tr = y[tr]
        
        # Use precomputed class weights from full dataset to avoid warm_start warnings
        est = _make_multiclass_estimator(cfg=cfg, family=family, class_weight=class_weight_dict)
        est.fit(np.asarray(X[tr], dtype=float), np.asarray(y_tr, dtype=int))
        proba = est.predict_proba(np.asarray(X[te], dtype=float))  # shape (n_te, n_classes)
        
        # Save fold model if requested
        fold_id = fold_idx + 1
        if save_models and model_dir and family:
            model_path = save_model(est, model_dir, family, "multiclass", fold=fold_id)
            saved_model_paths.append(model_path)
        
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

        fold_macro_auprc.append(macro_auprc)

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

    # Chance macro-AUPRC: mean of per-class prevalences
    prevalences = [(y == c).mean() for c in classes]
    chance_macro_auprc = float(np.mean(prevalences))

    # Permutation (Macro-AUPRC) with fixed splits
    rng = np.random.default_rng(42)
    perm_values = np.empty(n_perm, dtype=float)
    for i in tqdm(range(n_perm), desc=f"{family} multiclass permutation"):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        # Accumulate predictions across splits
        perm_true_all: List[int] = []
        perm_proba_all: List[np.ndarray] = []

        for (tr, te) in cv_splits:
            # Use precomputed class weights from full dataset to avoid warm_start warnings
            est = _make_multiclass_estimator(cfg=cfg, family=family, class_weight=class_weight_dict)
            est.fit(np.asarray(X[tr], dtype=float), np.asarray(y_perm[tr], dtype=int))
            proba = est.predict_proba(np.asarray(X[te], dtype=float))
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
        "n_folds": len(cv_splits),
        "n_perm": n_perm,
        "chance_macro_auprc": chance_macro_auprc,
    }

    # Fold metrics dataframe-style for saving
    fold_metrics = [
        {"MacroAUPRC": float(a)}
        for a in fold_macro_auprc
    ]

    scores_df = pd.DataFrame(mc_score_rows, columns = [
        "molecule_id", "family", "cv_fold", "true_lbl", "class_lbl", "score"
    ])
    
    # Train and save final model on full dataset if requested
    final_model_path = None
    if save_models and model_dir and family:
        final_model_path = save_final_models(
            X=X, y=y, model_dir=model_dir, family=family, 
            task_type="multiclass", feature_names=feature_names, cfg=cfg
        )

    return {
        "fold_metrics": fold_metrics,
        "perm_df": perm_df,
        "summary": summary,
        "scores": scores_df,
        "saved_models": {
            "fold_models": saved_model_paths,
            "final_model": final_model_path
        }
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
    """Write all required CSV and JSON outputs to disk.

    Args:
        out_dir: Target directory to write outputs.
        binary_results: Map family -> result dict from evaluate_binary_family().
        multiclass_results: Map family -> result dict from evaluate_multiclass_family().
        meta: Metadata dict to serialize as JSON.
    """
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
        for metric_name in ["AUPRC", "AUROC"]:
            m = summ[metric_name]
            bin_summary_rows.append(
                {
                    "family": family,
                    "metric": metric_name,
                    "mean": m.get("mean", np.nan),
                    "std": m.get("std", np.nan),
                    "ci_low": m.get("ci_low", np.nan),
                    "ci_high": m.get("ci_high", np.nan),
                    "observed": m.get("observed", np.nan),
                    "p_value": m.get("p_value", np.nan),
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
                    "AUROC": fm["AUROC"],
                }
            )

        # Confusion normalized
        cn = res["summary"].get("confusion_norm", {})
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
        for metric_name in ["MacroAUPRC"]:
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
                    "p_value": m.get("p_value", np.nan),
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
                }
            )

        pdf: pd.DataFrame = res["perm_df"].copy()
        pdf["family"] = family
        mc_perm_rows.append(pdf)

        scores: pd.DataFrame = res["scores"].copy()
        mc_scores_all.append(scores)
        
    # Write files
    pd.DataFrame(bin_summary_rows).to_csv(
        os.path.join(out_dir, "cRF_binary_summary.csv"), index=False
    )
    pd.DataFrame(bin_fold_rows).to_csv(
        os.path.join(out_dir, "cRF_binary_fold_metrics.csv"), index=False
    )
    pd.DataFrame(bin_confusion_rows).to_csv(
        os.path.join(out_dir, "cRF_binary_confusion_normalized.csv"), index=False
    )
    (pd.concat(bin_scores_rows, ignore_index=True) if bin_scores_rows else pd.DataFrame(columns=["molecule_id","family","y_true","y_score","cv_fold"]))\
        .to_csv(os.path.join(out_dir, "cRF_binary_scores.csv"), index=False)
    (pd.concat(bin_perm_rows, ignore_index=True) if bin_perm_rows else pd.DataFrame(columns=["family","perm_id","auprc_null","auroc_null"]))\
        .to_csv(os.path.join(out_dir, "cRF_binary_permutation_nulls.csv"), index=False)

    pd.DataFrame(mc_summary_rows).to_csv(
        os.path.join(out_dir, "cRF_multiclass_summary.csv"), index=False
    )
    pd.DataFrame(mc_fold_rows).to_csv(
        os.path.join(out_dir, "cRF_multiclass_fold_metrics.csv"), index=False
    )
    (pd.concat(mc_perm_rows, ignore_index=True) if mc_perm_rows else pd.DataFrame(columns=["family","perm_id","macro_auprc_null"]))\
        .to_csv(os.path.join(out_dir, "cRF_multiclass_permutation_macroauprc.csv"), index=False)

    (pd.concat(mc_scores_all, ignore_index=True) if mc_scores_all else pd.DataFrame(
        columns=["molecule_id", "family", "cv_fold", "true_lbl", "class_lbl", "score"])
    ).to_csv(os.path.join(out_dir, "cRF_multiclass_scores.csv"), index=False)

    # Meta JSON
    meta_path = os.path.join(out_dir, "cRF_meta.json")
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
    
    # Get pipeline information (using default family for metadata)
    binary_pipeline = _make_binary_pipeline(cfg=cfg)
    multiclass_estimator = _make_multiclass_estimator(cfg=cfg)
    
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
        "family_variance_thresholds": FAMILY_VARIANCE_THRESHOLDS,
        "pipeline": {
            "binary_pipeline": {
                "steps": pipeline_steps,
                "description": "Random Forest with balanced subsample class weights"
            },
            "multiclass_estimator": multiclass_info,
            "description": "One-vs-Rest classifier using Random Forest for multiclass classification"
        }
    }
    return meta


# no CLI integration: call main() from notebooks or scripts if desired
