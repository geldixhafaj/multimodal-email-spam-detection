
# seed_utils.py
# ===============================================================
# Unified seeding & determinism helpers for reproducible experiments.
# Works in CPU-only and CUDA environments, with best-effort fallbacks.
# ---------------------------------------------------------------
# Author: Geldi Xhafaj (MSc Dissertation)
# License: MIT (adjust to your repository policy)
# ===============================================================

from __future__ import annotations
import os
import random
import contextlib
from dataclasses import dataclass
from typing import Optional

# Optional imports guarded
try:
    import numpy as _np
except Exception:
    _np = None

try:
    import torch as _torch
except Exception:
    _torch = None


# ------------------------------
# Module-level global seed store
# ------------------------------
_GLOBAL_SEED: Optional[int] = None


def get_global_seed(default: int = 42) -> int:
    """Return the last seed set with set_global_seed/seed_everything (or default)."""
    return _GLOBAL_SEED if _GLOBAL_SEED is not None else default


def set_global_seed(seed: int) -> None:
    """Record the global seed without applying it (use seed_everything to apply)."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)


# ----------------------------------------------
# CUDA/cuBLAS/cuDNN determinism (if torch exists)
# ----------------------------------------------
def set_cublas_deterministic() -> None:
    """
    Configure cuBLAS workspace for deterministic reductions (if CUDA present).
    Only has effect in CUDA environments; safe no-op otherwise.
    """
    if _torch is None or not _torch.cuda.is_available():
        return
    # Accept either setting; consult PyTorch reproducibility docs.
    # If an env value already exists, leave it unchanged.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Two recommended options: ':4096:8' (smaller) or ':16:8' (bigger workspace)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_python_hash_seed(seed: int) -> None:
    """
    Ensure Python's hash-based operations are deterministic across runs.
    Must be set before interpreter starts to be fully effective.
    We still set it here for downstream subprocesses.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def seed_numpy(seed: int) -> None:
    if _np is not None:
        _np.random.seed(int(seed))


def seed_python(seed: int) -> None:
    random.seed(int(seed))


def seed_torch(seed: int, enforce_determinism: bool = True) -> None:
    if _torch is None:
        return
    s = int(seed)
    _torch.manual_seed(s)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(s)

    if enforce_determinism:
        # cuDNN: prefer determinism over speed
        try:
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
        except Exception:
            pass

        # Newer PyTorch flag for general determinism
        try:
            _torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older versions may not expose this API
            pass

        # cuBLAS workspace config (CUDA only)
        set_cublas_deterministic()


def seed_everything(seed: int = 42,
                    set_hash_seed: bool = True,
                    enforce_torch_determinism: bool = True) -> int:
    """
    Seed Python, NumPy, PyTorch (CPU/CUDA), and (optionally) Python hash seed.
    Returns the seed that was applied, and stores it as the global seed.

    Parameters
    ----------
    seed : int
        The global seed to apply.
    set_hash_seed : bool
        If True, sets PYTHONHASHSEED for current process & children.
    enforce_torch_determinism : bool
        If True, flips PyTorch deterministic flags & cuBLAS workspace.

    Notes
    -----
    - For perfect reproducibility across *different hardware*, determinism may
      still vary if you use different GPU models or library versions.
    - If you spawn DataLoader workers, use `build_worker_init_fn(seed)`.

    Returns
    -------
    int
        The applied seed.
    """
    s = int(seed)
    set_global_seed(s)
    if set_hash_seed:
        set_python_hash_seed(s)
    seed_python(s)
    seed_numpy(s)
    seed_torch(s, enforce_determinism=enforce_torch_determinism)
    return s


# -----------------------------------
# NumPy/Python temporary seeding block
# -----------------------------------
@dataclass
class _RNGState:
    py_state: object
    np_state: Optional[object] = None


@contextlib.contextmanager
def TempSeed(seed: int):
    """
    Temporarily set Python & NumPy RNG seeds inside a with-block, then restore.

    Example
    -------
    >>> from seed_utils import TempSeed
    >>> import numpy as np, random
    >>> a1 = np.random.rand()
    >>> with TempSeed(123):
    ...     a2 = np.random.rand()
    >>> with TempSeed(123):
    ...     a3 = np.random.rand()
    >>> assert a2 == a3  # identical inside same TempSeed
    """
    s = int(seed)
    # save states
    py_state = random.getstate()
    np_state = _np.random.get_state() if _np is not None else None
    try:
        random.seed(s)
        if _np is not None:
            _np.random.seed(s)
        yield
    finally:
        # restore states
        random.setstate(py_state)
        if _np is not None and np_state is not None:
            _np.random.set_state(np_state)


# ---------------------------------------------
# Deterministic DataLoader worker init (PyTorch)
# ---------------------------------------------
def torch_worker_init_fn(base_seed: int, worker_id: int) -> None:
    """
    Worker init function for PyTorch DataLoader.
    Ensures each worker has a deterministic, unique seed derived from base_seed.

    Usage (DataLoader)
    ------------------
    >>> from functools import partial
    >>> init_fn = partial(torch_worker_init_fn, base_seed=42)
    >>> DataLoader(dataset, num_workers=4, worker_init_fn=init_fn, generator=torch.Generator().manual_seed(42))
    """
    if _torch is None:
        return
    worker_seed = (int(base_seed) + worker_id) % (2**32 - 1)
    seed_python(worker_seed)
    seed_numpy(worker_seed)
    # For torch RNG used inside the worker
    _torch.manual_seed(worker_seed)


def build_worker_init_fn(base_seed: int):
    """
    Return a partial function suitable for DataLoader(worker_init_fn=...).

    Example
    -------
    >>> worker_init_fn = build_worker_init_fn(42)
    >>> DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
    """
    def _fn(worker_id: int):
        return torch_worker_init_fn(base_seed, worker_id)
    return _fn


# ---------------------------
# Friendly one-line reporter
# ---------------------------

