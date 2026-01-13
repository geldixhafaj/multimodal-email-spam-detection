
# metric_utils.py
# ===============================================================
# Metrics and plots for transformer baseline and multimodal MLP.
# Produces Accuracy, per-class Precision/Recall/F1, Macro/Micro-F1,
# ROC-AUC, confusion counts, threshold sweep, PR/ROC curves, and
# ready-made tables for Chapter 3.
# ---------------------------------------------------------------
# Author: Geldi Xhafaj (MSc Dissertation)
# License: MIT (adjust for your repo)
# ===============================================================

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# -----------------------
# Data container for metrics
# -----------------------
@dataclass
class BinaryMetrics:
    accuracy: float
    precision_spam: float
    recall_spam: float
    f1_spam: float
    precision_ham: float
    recall_ham: float
    f1_ham: float
    macro_f1: float
    micro_f1: float
    roc_auc: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

# -----------------------
# Confusion counts (TN, FP, FN, TP)
# -----------------------
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Return counts in the conventional layout:
      TN = cm[0,0], FP = cm[0,1], FN = cm[1,0], TP = cm[1,1]
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}

# -----------------------
# Metrics from hard predictions
# -----------------------
def binary_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> BinaryMetrics:
    """
    Compute binary metrics given hard predictions (0/1) and optional scores for ROC-AUC.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,), 0=Ham, 1=Spam
    y_pred : array-like of shape (n_samples,), 0/1 hard predictions
    y_score: optional probabilities for class 1 (Spam) to compute ROC-AUC

    Returns
    -------
    BinaryMetrics dataclass
    """
    acc = accuracy_score(y_true, y_pred)
    # per-class metrics: label=1 is Spam, label=0 is Ham
    p_spam, r_spam, f_spam, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    p_ham, r_ham, f_ham, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0], average="binary", zero_division=0
    )
    macro_f1 = np.mean([f_ham, f_spam])
    # micro metrics aggregate TP/FP/FN across classes

