import argparse
from DataLoader import DataLoader
from NaiveBayes import NaiveBayes
from EvaluationMetrics import EvaluationMetrics


def main() -> None:
    """
    Trains the Naive Bayes classifier, evaluates it, and logs the results.
    Loads data from SMSSpamCollection.txt, trains the model on 80% of the data,
    predicts on both training and testing sets, computes evaluation metrics,
    and writes a structured results log to results.log.
    """
    # Load and preprocess data
    x, y = DataLoader.load_data("SMSSpamCollection.txt")

    # Split into training and testing sets (80/20)
    x_train, x_test, y_train, y_test = DataLoader.split_data(x, y)

    # Train the model
    model = NaiveBayes()
    model.train(x_train, y_train)

    # Predict on both training and testing sets
    y_pred_train = model.prediction(x_train)
    y_pred_test  = model.prediction(x_test)

    # Compute evaluation metrics for both sets
    train_metrics = EvaluationMetrics()
    train_metrics.compute_metrics(y_train, y_pred_train)

    test_metrics = EvaluationMetrics()
    test_metrics.compute_metrics(y_test, y_pred_test)

    # Build log output
    lines = [
        "Naive Bayes Spam Classifier — Results Log",

        "\n--- Dataset ---",
        f"  Total samples    : {len(y)}",
        f"  Training samples : {len(y_train)}",
        f"  Testing samples  : {len(y_test)}",

        "\n--- Training Metrics ---",
        f"  Accuracy  : {train_metrics.accuracy:.4f}",
        f"  Precision : {train_metrics.precision:.4f}",
        f"  Recall    : {train_metrics.recall:.4f}",
        f"  F1 Score  : {train_metrics.f1:.4f}",
        f"  TP : {train_metrics.TP}",
        f"  TN : {train_metrics.TN}",
        f"  FP : {train_metrics.FP}",
        f"  FN : {train_metrics.FN}",

        "\n--- Testing Metrics ---",
        f"  Accuracy  : {test_metrics.accuracy:.4f}",
        f"  Precision : {test_metrics.precision:.4f}",
        f"  Recall    : {test_metrics.recall:.4f}",
        f"  F1 Score  : {test_metrics.f1:.4f}",
        f"  TP : {test_metrics.TP}",
        f"  TN : {test_metrics.TN}",
        f"  FP : {test_metrics.FP}",
        f"  FN : {test_metrics.FN}",

    ]

    # Print to console and save to results.log
    output = "\n".join(lines)
    print(output)

    with open("results.log", "w", encoding="utf-8") as f:
        f.write(output)


if __name__ == "__main__":
    main()