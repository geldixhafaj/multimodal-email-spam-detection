
"""
spamxai: Shared utilities for the dissertation codebase
=======================================================

This package initializer exposes a compact API for:
 - Project paths (Colab/local friendly) via PATHS and init_dirs()
 - Reproducible seeding across numpy / random / torch (if available)
 - Logging setup with console + optional file handler
 - Lightweight JSON / CSV I/O utilities
 - Device detection (CPU/CUDA/MPS) that degrades gracefully if torch is missing
 - Small timing helper and safe directory creation

Typical usage
-------------
>>> from spamxai import PATHS, init_dirs, get_logger, seed_everything, save_json
>>> init_dirs()                      # ensures outputs/ dirs exist
>>> log = get_logger("nb06")         # logger for Notebook 06
>>> seed_everything(42)              # reproducibility
>>> save_json({"hello":"world"}, PATHS["outputs_tables"] / "example.json")

Author: Geldi Xhafaj
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import json
import csv
import time
import random
import logging
from typing import Any, Dict, Iterable, Optional

# --------------------------
# Versioning & package info
# --------------------------
__package_name__ = "spamxai"
__version__ = "1.0.0"
__author__ = "Geldi Xhafaj"

# ------------------------------------------------
# Base directory detection (repo- and Colab-safe)
# ------------------------------------------------
# If running inside a repo, set base to the repository root (parent of package).
# If not sure, fall back to current working directory.
try:
    _HERE = Path(__file__).resolve()
    _PKG_ROOT = _HERE.parent  # .../spamxai
    BASE_DIR = _PKG_ROOT.parent  # repository root
except Exception:
    BASE_DIR = Path.cwd()

# ------------------------------------------
# Canonical relative folders used in thesis
# ------------------------------------------
# You can change these if your repo layout differs.
DATA_DIR              = BASE_DIR / "data"
MODELS_DIR            = BASE_DIR / "models"
OUTPUTS_BASELINE_DIR  = BASE_DIR / "outputs_baseline"
OUTPUTS_FEATURES_DIR  = BASE_DIR / "outputs_features"
OUTPUTS_MULTIMODAL_DIR= BASE_DIR / "outputs_multimodal"
OUTPUTS_XAI_DIR       = BASE_DIR / "outputs_xai"
OUTPUTS_TABLES_DIR    = BASE_DIR / "outputs_tables"
FIGURES_DIR           = BASE_DIR / "figures"
LOGS_DIR              = BASE_DIR / "logs"

# Expose a read-only-ish mapping for convenience
PATHS: Dict[str, Path] = {
    "base": BASE_DIR,
    "data": DATA_DIR,
    "models": MODELS_DIR,
    "outputs_baseline": OUTPUTS_BASELINE_DIR,
    "outputs_features": OUTPUTS_FEATURES_DIR,
    "outputs_multimodal": OUTPUTS_MULTIMODAL_DIR,
    "outputs_xai": OUTPUTS_XAI_DIR,
    "outputs_tables": OUTPUTS_TABLES_DIR,
    "figures": FIGURES_DIR,
    "logs": LOGS_DIR,
}

def init_dirs(extra: Optional[Iterable[Path]] = None) -> None:
    """
    Create all standard output directories (idempotent).
    Optionally create any extra paths you pass in.
    """
    for p in PATHS.values():
        if p is not None and p != BASE_DIR:
            p.mkdir(parents=True, exist_ok=True)
    if extra:
        for p in extra:
            Path(p).mkdir(parents=True, exist_ok=True)

# ----------------
# Logging helpers
# ----------------
def get_logger(name: str = __package_name__,
               level: int = logging.INFO,
               log_to_file: bool = True) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name : str
        Logger name (e.g., 'nb03', 'nb06').
    level : int
        Logging level (logging.INFO by default).
    log_to_file : bool
        If True, also write logs to logs/<name>.log

    Returns
    -------
    logging.Logger
    """
    init_dirs()  # ensure logs dir exists
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplication
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_to_file:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger

# -------------------------
# Reproducibility & device
# -------------------------
def seed_everything(seed: int = 42) -> None:
    """
    Seed python, numpy, and torch (if available).
    """
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not installed or not available; ignore
        pass

def get_device(prefer_cuda: bool = True):
    """
    Return a best-effort device object or string.
    - If torch available and CUDA enabled → 'cuda'
    - Else if torch available and MPS enabled → 'mps'
    - Else 'cpu'
    """
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        # macOS Metal
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return "cpu"  # generic string if torch is missing

# -------------
# I/O helpers
# -------------
def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:

