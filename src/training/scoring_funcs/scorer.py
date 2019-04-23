import abc


class Scorer(abc.ABC):
    """
    Represents a scoring criteria used to measure a model's performance.
    """

    @abc.abstractmethod
    def score(self, y_pred, y):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_better(self, current_score, other_score):
        raise NotImplementedError()
