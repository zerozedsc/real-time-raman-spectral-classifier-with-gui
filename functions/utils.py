import numpy as np


def translate_confusion_matrix(conf_matrix: np.ndarray, labels: list) -> dict:
    """
    Translate a confusion matrix into a dictionary with label names.

    Parameters
    ----------
    conf_matrix : np.ndarray
        The confusion matrix (square, shape [n_labels, n_labels]).
    labels : list
        List of label names, in the same order as used for the confusion matrix.

    Returns
    -------
    dict
        Dictionary with structure: {true_label: {predicted_label: count, ...}, ...}

    # Example usage:
    # conf_matrix = svc_data["confusion_matrix"]
    # labels = ["benign", "cancer"]
    # console_log(translate_confusion_matrix(conf_matrix, labels))
    """
    result = {}
    for i, true_label in enumerate(labels):
        result[true_label] = {}
        for j, pred_label in enumerate(labels):
            result[true_label][pred_label] = int(conf_matrix[i, j])
    return result
