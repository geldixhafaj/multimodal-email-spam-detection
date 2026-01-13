
# feature_utils.py
# ===============================================================
# Structural feature extraction for emails + train-only scaling.
# Produces a 13-dim feature vector per message, alignment checks,
# and writers for .npy + sidecar JSON metadata.
# ---------------------------------------------------------------
# Author: Geldi Xhafaj (MSc Dissertation)
# License: MIT (adjust for your repo)
# ===============================================================

from __future__ import annotations
from typing import List, Tuple, Dict, Any
import re
import json
import numpy as np
import pandas as pd

# --------------------- #
# Regexes / small utils #
# --------------------- #

_URL_RE = re.compile(r"(https?://[\w\-\.\?=&#/%+:~;,@!$'*()]+)", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"[\$€£¥]")
_PUNCT_RE = re.compile(r"[!\?\.,;:\-\(\)\[\]\{\}\*\_\~\'\"]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?)?\d{3}[\s-]?\d{4})")
_REPLY_FWD_RE = re.compile(r"^\s*(re|fw|fwd)\s*:", re.IGNORECASE)

def _char_count(text: str) -> int:
    return len(text or "")

def _alpha_count(text: str) -> int:
    return sum(ch.isalpha() for ch in (text or ""))

def _uppercase_ratio(text: str) -> float:
    a = _alpha_count(text)
    return (sum(ch.isupper() for ch in (text or "")) / a) if a else 0.0

def _digit_count(text: str) -> int:
    return sum(ch.isdigit() for ch in (text or ""))

def _url_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(_URL_RE.findall(text))

def _currency_flag(text: str) -> int:
    return int(bool(_CURRENCY_RE.search(text or "")))

def _punct_density(text: str) -> float:
    """Punctuation per 100 characters (robust to short strings)."""
    n = _char_count(text)
    if n == 0:
        return 0.0
    pct = len(_PUNCT_RE.findall(text)) / n
    return 100.0 * pct

def _html_presence(text: str) -> int:
    return int(bool(_HTML_TAG_RE.search(text or "")))

def _exclam_count(text: str) -> int:
    return (text or "").count("!")

def _quest_count(text: str) -> int:
    return (text or "").count("?")

def _at_sign_count(text: str) -> int:
    return (text or "").count("@")

def _repeated_char_ratio(text: str) -> float:
    """Ratio of characters that belong to 3+ runs, e.g., '!!!!!' or 'loooan'."""
    if not text:
        return 0.0
    runs = re.findall(r"(.)\1{2,}", text)
    return min(1.0, (sum(len(match) for match in runs) / max(1, len(text))))

def _number_token_ratio(text: str) -> float:
    """Proportion of tokens made solely of digits (e.g., codes, amounts)."""
    if not text:
        return 0.0
    toks = re.split(r"\s+", text.strip())
    if not toks:
        return 0.0
    num_only = sum(tok.isdigit() for tok in toks)
    return num_only / len(toks)

def _reply_forward_flag(text: str) -> int:
    return int(bool(_reply_forward_header(text)))

def _reply_forward_header(text: str) -> bool:
    return bool(_REPLY_FWD_RE.match(text or ""))

def _phone_flag(text: str) -> int:
    return int(bool(_PHONE_RE.search(text or "")))

# -------------------------------- #
# 13-feature extractor (RAW TEXT!) #
# -------------------------------- #

FEATURE_NAMES: List[str] = [
    # 6 features used throughout your dissertation:
    "url_count",               # int
    "uppercase_ratio",         # float [0,1]
    "digit_count",             # int
    "currency_symbol_flag",    # {0,1}
    "punct_density",           # float (per 100 chars)
    "html_presence",           # {0,1}
    # 7 additional robust cues to reach 13:
    "exclamation_count",       # int
    "question_count",          # int
    "at_sign_count",           # int
    "repeated_char_ratio",     # float [0,1]
    "number_token_ratio",      # float [0,1]
    "reply_forward_flag",      # {0,1}
    "phone_number_flag"        # {0,1}
]

def extract_struct_features(raw_texts: List[str]) -> np.ndarray:
    """
    Compute the 13-dim structural feature vector from RAW email texts.
    Returns: np.ndarray shape (N, 13) in the order of FEATURE_NAMES.
    """
    X = np.zeros((len(raw_texts), len(FEATURE_NAMES)), dtype=float)
    for i, t in enumerate(raw_texts):
        t = t or ""
        vals = [
            _url_count(t),
            _uppercase_ratio(t),
            _digit_count(t),
            _currency_flag(t),
            _punct_density(t),
            _html_presence(t),
            _exclam_count(t),
            _quest_count(t),
            _at_sign_count(t),
            _repeated_char_ratio(t),
            _number_token_ratio(t),
            _reply_forward_flag(t),
            _phone_flag(t),
        ]
        X[i, :] = vals
    return X

# ------------------------------------ #
# Train-only scaling and safe transform #
# ------------------------------------ #

def fit_standard_scaler(X_train: np.ndarray) -> Dict[str, Any]:
    """
    Fit StandardScaler-like stats on TRAIN ONLY.
    Returns dict with 'mean' and 'std' arrays (zeros guarded).
    """
    mu = X_train.mean(axis=0).astype(float)
    sd = X_train.std(axis=0, ddof=0).astype(float)
    sd[sd == 0] = 1.0
    return {"mean": mu.tolist(), "std": sd.tolist()}

def transform_with_stats(X: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    """Apply x' = (x - mean) / std using provided stats."""
    mu = np.asarray(stats["mean"], dtype=float)
    sd = np.asarray(stats["std"], dtype=float)
    return (X - mu) / sd

def extract_and_scale_from_frame(
    df_std: pd.DataFrame,
    stats: Dict[str, Any] = None,
    fit_on_train: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience for notebook use:
      - df_std must contain a 'raw_text' column (from data_utils.standardise_schema)
      - if fit_on_train=True → returns (X_scaled, stats_fit)
      - else → uses provided stats to transform
    """
    if "raw_text" not in df_std.columns:
        raise ValueError("df_std must contain 'raw_text' column for structural extraction.")
    X = extract_struct_features(df_std["raw_text"].tolist())
    if fit_on_train:
        stats = fit_standard_scaler(X)
        return transform_with_stats(X, stats), stats
    else:
        if stats is None:
            raise ValueError("stats is required when fit_on_train=False.")
        return transform_with_stats(X, stats), stats

# ---------------------- #
# Writers & consistency  #
# ---------------------- #

def save_struct_arrays(
    out_dir: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    feature_names: List[str] = None,
    scaling_stats: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Save structural matrices and sidecar metadata. Returns paths dict.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train_struct": f"{out_dir}/train_struct.npy",
        "val_struct":   f"{out_dir}/val_struct.npy",
        "test_struct":  f"{out_dir}/test_struct.npy",
        "train_labels": f"{out_dir}/train_labels.npy",
        "val_labels":   f"{out_dir}/val_labels.npy",
        "test_labels":  f"{out_dir}/test_labels.npy",
        "feature_names":f"{out_dir}/struct_feature_names.json",
        "scaling_stats":f"{out_dir}/struct_scaling_stats.json",
    }
    np.save(paths["train_struct"], X_train)
    np.save(paths["val_struct"],   X_val)
    np.save(paths["test_struct"],  X_test)

