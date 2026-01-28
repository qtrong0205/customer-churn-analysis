from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(y_true, y_pred):
    """
    Evaluate classification model performance.

    Parameters:
    - y_true: ground truth labels
    - y_pred: predicted labels

    Returns:
    - accuracy (float)
    """

    accuracy = accuracy_score(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return accuracy
