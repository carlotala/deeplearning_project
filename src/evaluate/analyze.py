"""
src/evaluate/analyze.py

Analysis utilities for classification models.
- Computes metrics: accuracy, F1, precision, recall, sensitivity, specificity, AUC, confusion matrix
- Plots: ROC curve, confusion matrix, per-class metrics
- Designed for multi-class image classification
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
from src.utils import plots


def compute_metrics(y_true, y_pred, y_prob=None, average="macro", class_names=None):
    """
    Compute common classification metrics for multi-class problems.
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted class indices.
        y_prob (array-like, optional): Predicted probabilities (n_samples, n_classes).
        average (str): Averaging method for multi-class metrics.
        class_names (list, optional): List of class names.
    Returns:
        dict: Dictionary of metrics.
    """
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, average=average)
    metrics["precision"] = precision_score(y_true, y_pred, average=average)
    metrics["recall"] = recall_score(y_true, y_pred, average=average)
    metrics["specificity"] = compute_specificity(y_true, y_pred, average=average)
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
        except Exception:
            metrics["auc"] = None
    else:
        metrics["auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, target_names=class_names)
    return metrics


def compute_specificity(y_true, y_pred, average="macro"):
    """
    Compute specificity for multi-class classification.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specificity_per_class = []
    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(np.delete(cm, i, axis=0)[:, i])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)
    if average == "macro":
        return np.mean(specificity_per_class)
    elif average == "none":
        return specificity_per_class
    else:
        raise ValueError(f"Unknown average: {average}")


def plot_confusion_matrix(cm, class_names, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    Plot confusion matrix using matplotlib.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob, class_names):
    """
    Plot ROC curve for each class and compute AUC.
    """
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def label_binarize(y, classes):
    """
    Binarize labels in a one-vs-all fashion for ROC curve plotting.
    """
    y = np.array(y)
    Y = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        Y[:, i] = (y == c).astype(int)
    return Y


def analyze_classification(y_true, y_pred, y_prob=None, class_names=None, average="macro", plot=True):
    """
    Main function to analyze classification results.
    Args:
        y_true: Ground truth labels (list or np.array)
        y_pred: Predicted labels (list or np.array)
        y_prob: Predicted probabilities (n_samples, n_classes), optional
        class_names: List of class names
        average: Averaging method for metrics
        plot: Whether to plot confusion matrix and ROC
    Returns:
        metrics: dict of computed metrics
    """
    metrics = compute_metrics(y_true, y_pred, y_prob, average=average, class_names=class_names)
    print("\nClassification Report:\n", metrics["classification_report"])
    if plot and class_names is not None:
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, normalize=True)
        if y_prob is not None:
            plot_roc_curve(y_true, y_prob, class_names)
    return metrics
