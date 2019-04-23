import abc


class Search(abc.ABC):

    def __init__(self, model_factory, param_values, scoring_func, train_loader, validation_loader):
        """
        Represents a searching algorithm for tuning a model.

        :param model_factory: A function that returns a new model instance given hyper params
        :param param_values: A dict containing model and training hyper params and their possible values.
        :param scoring_func: A function used to measure a model's performance.
        :param train_loader: A loader that loads the training data.
        :param validation_loader: A loader that loads the validation data.
        """
        self.model_factory = model_factory
        self.param_values = param_values
        self.scoring_func = scoring_func
        self.train_loader = train_loader
        self.validation_loader = validation_loader

    @abc.abstractmethod
    def get_tuned_model(self):
        raise NotImplementedError()
