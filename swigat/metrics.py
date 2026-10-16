import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def classification_metrics(labels, probabilities, num_classes):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(auc),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
    }


def weighted_prf(labels, probabilities):
    labels = np.asarray(labels)
    predictions = np.asarray(probabilities).argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return float(precision), float(recall), float(f1)

