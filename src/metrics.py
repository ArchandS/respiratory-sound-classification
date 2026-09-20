"""
Evaluation metrics.

Why we do NOT select models on raw accuracy
-------------------------------------------
The dataset is 52.8% 'Normal'. A model that predicts 'Normal' for everything
scores 52.8% accuracy while being clinically useless — it detects zero disease.
Accuracy actively rewards ignoring the rare Wheeze and Both classes, which are
the ones we care about.

We therefore report and select on the ICBHI Score, the official challenge
metric, alongside macro-F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, classification_report,
    balanced_accuracy_score,
)


def icbhi_score(y_true, y_pred, normal_class: int = 0):
    """
    Official ICBHI 2017 metric.

        Specificity (Sp) = correctly identified Normal / all Normal
        Sensitivity (Se) = correctly identified Abnormal / all Abnormal
                           (abnormal = Crackle + Wheeze + Both, pooled)
        Score            = (Sp + Se) / 2

    Returns (score, sensitivity, specificity).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    normal_mask = (y_true == normal_class)
    abnormal_mask = ~normal_mask

    n_normal = normal_mask.sum()
    n_abnormal = abnormal_mask.sum()

    # Specificity: Normal cycles predicted as Normal
    sp = ((y_pred[normal_mask] == normal_class).sum() / n_normal
          if n_normal > 0 else 0.0)

    # Sensitivity: abnormal cycles given their CORRECT abnormal class.
    # (The strict ICBHI definition requires the exact class, not merely
    # "predicted something abnormal".)
    se = ((y_pred[abnormal_mask] == y_true[abnormal_mask]).sum() / n_abnormal
          if n_abnormal > 0 else 0.0)

    return float((sp + se) / 2), float(se), float(sp)


def compute_all(y_true, y_pred, class_names):
    score, se, sp = icbhi_score(y_true, y_pred)
    return {
        "icbhi_score": score,
        "sensitivity": se,
        "specificity": sp,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def format_metrics(m) -> str:
    return (f"ICBHI={m['icbhi_score']:.4f} "
            f"(Se={m['sensitivity']:.3f} Sp={m['specificity']:.3f}) | "
            f"mF1={m['macro_f1']:.4f} | acc={m['accuracy']*100:.2f}%")


def full_report(y_true, y_pred, class_names) -> str:
    m = compute_all(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    lines = []
    lines.append("=" * 62)
    lines.append("EVALUATION")
    lines.append("=" * 62)
    lines.append(f"ICBHI Score       : {m['icbhi_score']:.4f}")
    lines.append(f"  Sensitivity (Se): {m['sensitivity']:.4f}")
    lines.append(f"  Specificity (Sp): {m['specificity']:.4f}")
    lines.append(f"Macro F1          : {m['macro_f1']:.4f}")
    lines.append(f"Balanced accuracy : {m['balanced_accuracy']:.4f}")
    lines.append(f"Raw accuracy      : {m['accuracy']*100:.2f}%")

    majority = np.bincount(np.asarray(y_true)).max() / len(y_true)
    lines.append(f"(majority-class baseline accuracy: {majority*100:.2f}%)")
    lines.append("")
    lines.append(classification_report(y_true, y_pred,
                                       target_names=class_names,
                                       zero_division=0, digits=4))
    lines.append("Confusion matrix (rows = true, cols = predicted)")
    header = " " * 10 + "".join(f"{c[:7]:>9}" for c in class_names)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"{class_names[i][:9]:<10}" +
                     "".join(f"{v:>9d}" for v in row))
    lines.append("=" * 62)
    return "\n".join(lines)
