class EvaluationMetrics:
    def __init__(self):
        """Initializes the EvaluationMetrics class.

        Attributes:
            TP (int): True positives  — predicted spam, actually spam.
            TN (int): True negatives  — predicted ham,  actually ham.
            FP (int): False positives — predicted spam, actually ham.
            FN (int): False negatives — predicted ham,  actually spam.
            accuracy (float):  Fraction of all predictions that were correct.
            precision (float): Fraction of spam predictions that were actually spam.
            recall (float):    Fraction of actual spam that was correctly identified.
            f1 (float):        Harmonic mean of precision and recall.
        """
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.accuracy  = 0.0
        self.precision = 0.0
        self.recall    = 0.0
        self.f1        = 0.0

    def compute_metrics(self, y_true: list[int], y_pred: list[int]) -> dict:
        """Computes evaluation metrics such as accuracy, precision, recall, and F1-score.

        Args:
            y_true (list[int]): Ground truth labels (1 = spam, 0 = ham).
            y_pred (list[int]): Predicted labels (1 = spam, 0 = ham).

        Returns:
            dict: TP, TN, FP, FN, accuracy, precision, recall, and f1.
        """
        # Count TP, TN, FP, FN in a single pass
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                self.TP += 1       # correctly identified spam
            elif true == 0 and pred == 0:
                self.TN += 1       # correctly identified ham
            elif true == 0 and pred == 1:
                self.FP += 1       # ham wrongly flagged as spam
            else:
                self.FN += 1       # spam missed, labelled as ham

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        self.accuracy = (self.TP + self.TN) / len(y_true)

        # Precision = TP / (TP + FP)
        self.precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0

        # Recall = TP / (TP + FN)
        self.recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0.0

        return {
            "TP":        self.TP,
            "TN":        self.TN,
            "FP":        self.FP,
            "FN":        self.FN,
            "accuracy":  self.accuracy,
            "precision": self.precision,
            "recall":    self.recall,
            "f1":        self.f1,
        }