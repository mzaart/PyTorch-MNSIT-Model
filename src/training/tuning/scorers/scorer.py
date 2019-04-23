import abc


class Scorer(abc.ABC):

    def __init__(self, model, hyper_params, scoring_func, train_loader, validation_loader):
        """
        This class measures the model's performance on a given data set.

        :param model: The model to be trained and scored.
        :param hyper_params: A dict containing training parameter values
        :param scoring_func: The scoring function used to measure the model's performance
        :param train_loader: A loader that loads the training data.
        :param validation_loader: A loader that loads the validation data.
        """
        self.hyper_params = hyper_params
        self.model = model
        self.scoring_func = scoring_func
        self.train_loader = train_loader
        self.validation_loader = validation_loader

    @abc.abstractmethod
    def score(self):
        raise NotImplementedError()
