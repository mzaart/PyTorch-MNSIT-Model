import random
from .parameters_domain import ParametersDomain


class RandomParametersDomain(ParametersDomain):
    """
    This class fetches parameters from the domain in a random fashion.
    """

    def __init__(self, param_values):
        super().__init__(param_values)

    def __next__(self):
        values = {}
        for param in self.param_values:
            rand_index = random.randint(0, len(self.param_values[param]) - 1)
            values[param] = self.param_values[param][rand_index]
        return values
