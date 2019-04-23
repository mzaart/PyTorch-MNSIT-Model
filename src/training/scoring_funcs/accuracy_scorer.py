from .scorer import Scorer
from sklearn.metrics.scorer import accuracy_score


class AccuracyScorer(Scorer):

    def score(self, y_pred, y):
        return accuracy_score(y.numpy(), y_pred.numpy(), normalize=True)

    def is_better(self, current_mse_score, other_mse_score):
        return current_mse_score > other_mse_score
