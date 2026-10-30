"""
evaluation.py - metric computation and ALL plotting helpers.

Plot functions return matplotlib Figure objects built with `Figure()` (not
pyplot) so they are safe inside Streamlit and never open pop-up windows.
"""
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

TEAL, NAVY, SKY, GREEN, CORAL = "#0F9D9A", "#0B5C63", "#8ED1DC", "#4CAF93", "#E57373"
MODEL_COLORS = {"Logistic Regression": TEAL, "Random Forest": SKY}
CLASS_NAMES = ["Non-diabetic", "Diabetic"]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """All metrics come from the real predictions - nothing is hard-coded."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
    }


def plot_confusion_matrix(cm, title="Confusion Matrix") -> Figure:
    fig = Figure(figsize=(4.6, 3.8))
    ax = fig.subplots()
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="GnBu", cbar=False, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 14})
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, color=NAVY)
    fig.tight_layout()
    return fig


def plot_model_comparison(results: dict) -> Figure:
    """results = {model_name: test_metrics_dict}"""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fig = Figure(figsize=(8, 4))
    ax = fig.subplots()
    width = 0.8 / max(len(results), 1)
    for i, (name, m) in enumerate(results.items()):
        pos = np.arange(len(keys)) + i * width
        vals = [m[k] for k in keys]
        bars = ax.bar(pos, vals, width, label=name, color=MODEL_COLORS.get(name, GREEN))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(keys)) + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score (held-out test set)")
    ax.set_title("Model comparison", color=NAVY); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_roc_curves(roc_data: dict) -> Figure:
    """roc_data = {model_name: {'fpr': [...], 'tpr': [...], 'auc': float}}"""
    fig = Figure(figsize=(5.2, 4.2))
    ax = fig.subplots()
    for name, d in roc_data.items():
        ax.plot(d["fpr"], d["tpr"], lw=2, color=MODEL_COLORS.get(name, GREEN),
                label=f"{name} (AUC = {d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random guess")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves", color=NAVY); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_importance(importances: dict, title="Random Forest feature importance") -> Figure:
    items = sorted(importances.items(), key=lambda kv: kv[1])
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.barh([k for k, _ in items], [v for _, v in items], color=TEAL)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title(title, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_probability(prob_positive: float) -> Figure:
    """Horizontal bar showing the model's estimated class probabilities."""
    fig = Figure(figsize=(6, 1.9))
    ax = fig.subplots()
    probs = [1 - prob_positive, prob_positive]
    bars = ax.barh(CLASS_NAMES, probs, color=[GREEN, TEAL], height=0.55)
    for b, p in zip(bars, probs):
        ax.text(min(p + 0.02, 0.86), b.get_y() + b.get_height() / 2, f"{p * 100:.1f}%", va="center",
                fontweight="bold", color=NAVY)
    ax.set_xlim(0, 1); ax.set_xlabel("Model-estimated probability")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_class_distribution(class_counts: dict) -> Figure:
    fig = Figure(figsize=(4.6, 3.6))
    ax = fig.subplots()
    labels = [CLASS_NAMES[int(k)] for k in class_counts]
    vals = list(class_counts.values())
    bars = ax.bar(labels, vals, color=[GREEN, TEAL])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center")
    ax.set_ylabel("Records"); ax.set_title("Target class distribution", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df) -> Figure:
    fig = Figure(figsize=(7.5, 6))
    ax = fig.subplots()
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="GnBu", square=True, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    ax.set_title("Correlation heatmap", color=NAVY)
    fig.tight_layout()
    return fig


def plot_feature_distribution(df, feature, target="Outcome") -> Figure:
    """Histogram of one feature, split by class. Zeros in medical columns are hidden."""
    from src.preprocessing import ZERO_AS_MISSING_COLUMNS
    data = df[[feature, target]].copy()
    if feature in ZERO_AS_MISSING_COLUMNS:
        data = data[data[feature] != 0]
    fig = Figure(figsize=(6, 3.8))
    ax = fig.subplots()
    sns.histplot(data=data, x=feature, hue=target, bins=25, kde=True,
                 palette={0: GREEN, 1: TEAL}, ax=ax)
    ax.set_title(f"Distribution of {feature}", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
