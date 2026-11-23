"""Metric calculation and result plots, shared by every model."""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from src import config


def evaluate(pipeline, X_test, y_test, labels):
    """Score a fitted pipeline on held-out data; returns a JSON-friendly dict."""
    t0 = time.perf_counter()
    pred = pipeline.predict(X_test)
    predict_seconds = time.perf_counter() - t0

    out = {"accuracy": float(accuracy_score(y_test, pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred, average=avg, labels=labels, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = map(float, (p, r, f))
    out["per_class"] = classification_report(
        y_test, pred, labels=labels, output_dict=True, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_test, pred, labels=labels).tolist()
    out["predict_seconds_total"] = predict_seconds
    out["predict_ms_per_row"] = 1000 * predict_seconds / max(len(X_test), 1)
    return out


def plot_confusion_matrix(cm, labels, title, filename):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1 + 0.8 * len(labels), 1 + 0.7 * len(labels)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)


def plot_model_comparison(results: dict, filename="model_comparison.png"):
    """Grouped bars: macro-F1 on the random test split vs. unseen-merchant CV."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results)
    random_f1 = [results[n]["test"]["f1_macro"] for n in names]
    grouped_f1 = [results[n]["unseen_merchant_cv"]["f1_macro_mean"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, random_f1, 0.4, label="Random test split", color="#3b6ea5")
    ax.bar(x + 0.2, grouped_f1, 0.4, label="Unseen-merchant CV", color="#b5533c")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)
