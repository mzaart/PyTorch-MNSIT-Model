import abc


class ParametersDomain(abc.ABC):
    """
    Represents the domain where the model parameters live.
    This class is meant to be used as an iterator where calling next() returns a new batch of
    parameters from the domain.
    """

    def __init__(self, param_values):
        self.param_values = param_values

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError()