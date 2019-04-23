from .scorer import Scorer
from torch.nn import MSELoss


class MSEScorer(Scorer):

    def __init__(self):
        self.loss = MSELoss()

    def score(self, y_pred, y):
        self.loss(y_pred, y)

    def is_better(self, current_mse_score, other_mse_score):
        return current_mse_score < other_mse_score
