
# viz_utils.py
# ===============================================================
# Publication-quality visualizations for Chapter 3 results/XAI:
# - Confusion matrices, PR/ROC overlays
# - Metric bars and Δ bars
# - Error counts table-as-figure
# - SHAP visuals: global grouped, structural beeswarm, matrix heatmap, local bars
# - LIME local bars
# - Multi-panel dashboards (results + XAI)
# ---------------------------------------------------------------
# Author: Geldi Xhafaj (MSc Dissertation)
# License: MIT (adjust to your repository policy)
# ===============================================================

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Global styling & palette
# ---------------------------
PALETTE = {
    "baseline": "#4C78A8",   # blue
    "multimodal": "#F58518", # orange
    "positive": "#2CA02C",   # green
    "negative": "#D62728",   # red
    "embedding": "#4C78A8",  # blue for text embeddings
    "structural": "#F58518", # orange for structural features
}

def set_theme(context: str = "talk", grid: bool = True, font_size: int = 12, dpi: int = 140, save_dpi: int = 220):
    """
    Apply a consistent theme across figures. Call once at notebook start.
    """
    sns.set(style="whitegrid" if grid else "white", context=context)
    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["savefig.dpi"] = save_dpi
    mpl.rcParams["axes.titlesize"] = font_size + 2
    mpl.rcParams["axes.labelsize"] = font_size
    mpl.rcParams["xtick.labelsize"] = font_size - 1
    mpl.rcParams["ytick.labelsize"] = font_size - 1
    mpl.rcParams["legend.fontsize"] = font_size - 1

def _ensure_dir(save_path: Optional[str]):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

def _annotate_bars(ax, values: List[float], xs: List[float], dy: float = 0.0008, fmt: str = "{:.3f}"):
    for x, v in zip(xs, values):
        ax.text(x, v + (dy if v >= 0 else -dy), fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=max(9, int(mpl.rcParams["font.size"])-1))

# ---------------------------
# Confusion matrix (heatmap)
# ---------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: Tuple[str, str] = ("Ham", "Spam"),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    cm shape must be (2,2) with layout:
      [[TN, FP],
       [FN, TP]]
    """
    _ensure_dir(save_path)
    fig = plt.figure(figsize=(5.6, 4.8))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=labels, yticklabels=labels,
                     linewidths=.5, linecolor="white")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# PR overlay for Baseline vs Multimodal (Spam)
# ---------------------------
def plot_pr_overlay(
    baseline_pr: Tuple[np.ndarray, np.ndarray],   # (precision, recall)
    multimodal_pr: Tuple[np.ndarray, np.ndarray], # (precision, recall)
    title: str = "Precision–Recall (Spam): Baseline vs Multimodal",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    precision-recall arrays as returned by sklearn.metrics.precision_recall_curve.
    """
    _ensure_dir(save_path)
    p_b, r_b = baseline_pr
    p_m, r_m = multimodal_pr
    fig = plt.figure(figsize=(6.8, 4.8))
    plt.plot(r_b, p_b, color=PALETTE["baseline"], lw=2, label="Baseline")
    plt.plot(r_m, p_m, color=PALETTE["multimodal"], lw=2, label="Multimodal")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# ROC overlay for Baseline vs Multimodal (Spam)
# ---------------------------
def plot_roc_overlay(
    baseline_roc: Tuple[np.ndarray, np.ndarray],   # (fpr, tpr)
    multimodal_roc: Tuple[np.ndarray, np.ndarray], # (fpr, tpr)
    auc_baseline: Optional[float] = None,
    auc_multimodal: Optional[float] = None,
    title: str = "ROC (Spam): Baseline vs Multimodal",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    fpr_b, tpr_b = baseline_roc
    fpr_m, tpr_m = multimodal_roc
    fig = plt.figure(figsize=(6.8, 4.8))
    label_b = f"Baseline (AUC={auc_baseline:.3f})" if auc_baseline is not None else "Baseline"
    label_m = f"Multimodal (AUC={auc_multimodal:.3f})" if auc_multimodal is not None else "Multimodal"
    plt.plot(fpr_b, tpr_b, color=PALETTE["baseline"], lw=2, label=label_b)
    plt.plot(fpr_m, tpr_m, color=PALETTE["multimodal"], lw=2, label=label_m)
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# Metric bars (Baseline vs Multimodal)
# ---------------------------
def plot_metric_bars(
    metrics: List[str],
    baseline_vals: List[float],
    multimodal_vals: List[float],
    ylim: Tuple[float, float] = (0.90, 1.00),
    title: str = "Model Performance Comparison (Test)",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    x = np.arange(len(metrics))
    width = 0.36
    fig = plt.figure(figsize=(10.0, 4.8))
    ax = plt.gca()
    ax.bar(x - width/2, baseline_vals, width, label="Baseline", color=PALETTE["baseline"])
    ax.bar(x + width/2, multimodal_vals, width, label="Multimodal", color=PALETTE["multimodal"])
    ax.set_xticks(x, metrics, rotation=20, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    # Annotate
    _annotate_bars(ax, baseline_vals, x - width/2)
    _annotate_bars(ax, multimodal_vals, x + width/2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# Δ bars (Multimodal − Baseline)
# ---------------------------
def plot_delta_bars(
    metrics: List[str],
    baseline_vals: List[float],
    multimodal_vals: List[float],
    title: str = "Metric Improvements from Multimodal Fusion (Test)",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    deltas = np.array(multimodal_vals) - np.array(baseline_vals)
    colors = [PALETTE["positive"] if d >= 0 else PALETTE["negative"] for d in deltas]
    fig = plt.figure(figsize=(9.0, 4.2))
    ax = plt.gca()
    ax.bar(metrics, deltas, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.0008 if d >= 0 else -0.0008), f"{d:+.4f}",
                ha="center", va="bottom" if d >= 0 else "top", fontsize=max(9, int(mpl.rcParams["font.size"])-1))
    ax.set_ylabel("Δ (Multimodal − Baseline)")
    ax.set_title(title)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# Error counts table as figure
# ---------------------------
def plot_error_counts_table(
    baseline_counts: Dict[str, int],
    multimodal_counts: Dict[str, int],
    title: str = "Error Counts (Test)",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    rows = ["Baseline", "Multimodal"]
    vals = np.array([
        [baseline_counts["TN"], baseline_counts["FP"], baseline_counts["FN"], baseline_counts["TP"]],
        [multimodal_counts["TN"], multimodal_counts["FP"], multimodal_counts["FN"], multimodal_counts["TP"]],
    ])
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    ax.axis("off")
    col_labels = ["TN", "FP", "FN", "TP"]
    table = ax.table(cellText=vals, rowLabels=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if key[0] == 0 or key[1] == -1:
            cell.set_text_props(weight="bold")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# SHAP: global grouped bar
# ---------------------------
def plot_shap_global_grouped(
    group_names: List[str],     # ["TextEmbedding(total)", "uppercase_ratio", ...]
    group_values: List[float],  # mean |SHAP|
    title: str = "Global SHAP Importance: TextEmbedding vs Structural Features",
    xlabel: str = "Mean |SHAP|",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    colors = [PALETTE["embedding"]] + [PALETTE["structural"]] * (len(group_names) - 1)
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = sns.barplot(x=group_values, y=group_names, palette=colors, orient="h")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# SHAP: structural beeswarm (jittered scatter)
# ---------------------------
def plot_shap_struct_beeswarm(
    shap_struct_values: np.ndarray,   # shape (n_samples, n_struct_features)
    struct_feature_names: List[str],
    title: str = "Structural Features: SHAP Beeswarm (test subset)",
    xlabel: str = "SHAP value",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = plt.gca()
    n_samples, n_feats = shap_struct_values.shape
    for i in range(n_feats):
        y = np.full(n_samples, i) + np.random.normal(0, 0.05, size=n_samples)
        x = shap_struct_values[:, i]
        ax.scatter(x, y, s=12, alpha=0.5, color=PALETTE["structural"])
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(struct_feature_names)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# SHAP: values matrix heatmap (struct-only)
# ---------------------------
def plot_shap_matrix(
    shap_struct_values: np.ndarray,   # shape (n_samples, n_struct_features)
    title: str = "SHAP Values Matrix – Structural Features",
    save_path: Optional[str] = None
) -> plt.Figure:
    _ensure_dir(save_path)
    fig = plt.figure(figsize=(8.8, 5.3))
    ax = sns.heatmap(shap_struct_values, cmap="RdBu_r", center=0,
                     cbar_kws={"label": "SHAP value"})
    ax.set_xlabel("Structural features")
    ax.set_ylabel("Test instances (subset)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# SHAP/LIME local bars
# ---------------------------
def plot_local_bar(
    names: List[str],
    values: np.ndarray,
    title: str,
    xlabel: str = "Contribution",
    save_path: Optional[str] = None,
    positive_color: str = PALETTE["embedding"],
    negative_color: str = PALETTE["structural"],
    top_k: int = 15
) -> plt.Figure:
    """
    Plot top-K local contributions (SHAP or LIME). Expects signed values.
    """
    _ensure_dir(save_path)
    order = np.argsort(-np.abs(values))
    names_sorted = [names[i] for i in order][:top_k]
    vals_sorted = values[order][:top_k]
    colors = [positive_color if v >= 0 else negative_color for v in vals_sorted]
    fig = plt.figure(figsize=(10.5, 6.0))
    ax = plt.gca()
    ax.barh(names_sorted, vals_sorted, color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

# ---------------------------
# Dashboards (multi-panel)
# ---------------------------
def make_results_dashboard(
    cm_baseline: np.ndarray,
    cm_multimodal: np.ndarray,
    metrics: List[str],
    baseline_vals: List[float],
    multimodal_vals: List[float],
    deltas_title: str = "Metric Improvements from Multimodal Fusion (Test)",
    fig_size: Tuple[float, float] = (12.5, 8.5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    2×2 grid:
      [0,0] baseline CM
      [0,1] multimodal CM
      [1,0] metric bars
      [1,1] Δ bars
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    # baseline CM
    sns.heatmap(cm_baseline, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=("Ham","Spam"), yticklabels=("Ham","Spam"),
                linewidths=.5, linecolor="white", ax=axes[0,0])
    axes[0,0].set_title("Baseline – Confusion Matrix")
    axes[0,0].set_xlabel("Predicted")
    axes[0,0].set_ylabel("True")
    # multimodal CM
    sns.heatmap(cm_multimodal, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=("Ham","Spam"), yticklabels=("Ham","Spam"),
                linewidths=.5, linecolor="white", ax=axes[0,1])
    axes[0,1].set_title("Multimodal – Confusion Matrix")
    axes[0,1].set_xlabel("Predicted")
    axes[0,1].set_ylabel("True")
    # metric bars
    x = np.arange(len(metrics)); width = 0.36
    axes[1,0].bar(x - width/2, baseline_vals, width, label="Baseline", color=PALETTE["baseline"])
    axes[1,0].bar(x + width/2, multimodal_vals, width, label="Multimodal", color=PALETTE["multimodal"])
    axes[1,0].set_xticks(x, metrics, rotation=20, ha="right")
    axes[1,0].set_ylim(0.90, 1.00)
    axes[1,0].set_ylabel("Score")
    axes[1,0].set_title("Performance Comparison")
    axes[1,0].legend()
    # Δ bars
    deltas = np.array(multimodal_vals) - np.array(baseline_vals)
    colors = [PALETTE["positive"] if d >= 0 else PALETTE["negative"] for d in deltas]
    axes[1,1].bar(metrics, deltas, color=colors)
    axes[1,1].axhline(0, color="k", linewidth=0.8)
    axes[1,1].set_ylabel("Δ (Multimodal − Baseline)")
    axes[1,1].set_title(deltas_title)
    axes[1,1].set_xticklabels(metrics, rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def make_xai_dashboard(
    shap_group_names: List[str],
    shap_group_values: List[float],
    shap_struct_values: np.ndarray,        # (n_samples, n_struct_features)
    struct_feature_names: List[str],
    local_names_tp: List[str], local_vals_tp: np.ndarray,
    local_names_err: List[str], local_vals_err: np.ndarray,
    fig_size: Tuple[float, float] = (14.0, 9.5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    2×2 XAI dashboard:
      [0,0] SHAP global grouped
      [0,1] Structural beeswarm
      [1,0] Local SHAP/LIME (TP)
      [1,1] Local SHAP/LIME (Error)
    """
    _ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    # Global grouped
    colors = [PALETTE["embedding"]] + [PALETTE["structural"]] * (len(shap_group_names) - 1)
    sns.barplot(x=shap_group_values, y=shap_group_names, palette=colors, orient="h", ax=axes[0,0])
    axes[0,0].set_title("Global SHAP Importance")
    axes[0,0].set_xlabel("Mean |SHAP|")
    # Beeswarm (jittered)
    n_samples, n_feats = shap_struct_values.shape
    for i in range(n_feats):
        y = np.full(n_samples, i) + np.random.normal(0, 0.05, size=n_samples)
        x = shap_struct_values[:, i]
        axes[0,1].scatter(x, y, s=12, alpha=0.5, color=PALETTE["structural"])
    axes[0,1].set_yticks(range(n_feats))
    axes[0,1].set_yticklabels(struct_feature_names)
    axes[0,1].set_title("Structural Beeswarm")
    axes[0,1].set_xlabel("SHAP value")
    # Local TP
    order_tp = np.argsort(-np.abs(local_vals_tp))
    names_tp = [local_names_tp[i] for i in order_tp][:15]
    vals_tp  = local_vals_tp[order_tp][:15]
    colors_tp = [PALETTE["embedding"] if v >= 0 else PALETTE["structural"] for v in vals_tp]
    axes[1,0].barh(names_tp, vals_tp, color=colors_tp)
    axes[1,0].axvline(0, color="k", linewidth=0.8)
    axes[1,0].set_title("Local Explanation (TP)")
    axes[1,0].set_xlabel("Contribution")
    # Local ERR
    order_err = np.argsort(-np.abs(local_vals_err))
    names_err = [local_names_err[i] for i in order_err][:15]
    vals_err  = local_vals_err[order_err][:15]
    colors_err = [PALETTE["embedding"] if v >= 0 else PALETTE["structural"] for v in vals_err]
    axes[1,1].barh(names_err, vals_err, color=colors_err)
    axes[1,1].axvline(0, color="k", linewidth=0.8)
    axes[1,1].set_title("Local Explanation (Error)")
    axes[1,1].set_xlabel("Contribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

