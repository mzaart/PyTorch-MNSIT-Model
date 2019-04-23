from .scorer import Scorer
from sklearn.metrics.scorer import f1_score


class FMeasureScorer(Scorer):

    def score(self, y_pred, y):
        return f1_score(y.numpy(), y_pred.numpy(), average='micro')

    def is_better(self, current_mse_score, other_mse_score):
        return current_mse_score > other_mse_score
