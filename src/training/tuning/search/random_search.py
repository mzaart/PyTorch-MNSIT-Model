from .search import Search
from ..parameters_domain import RandomParametersDomain
from ..scorers import DenseNetworkScorer


class RandomSearch(Search):
    """
    Represents the Random Search algorithm.
    """

    def __init__(self, num_iterations, model_factory, param_values, scoring_func, train_loader, validation_loader):
        super().__init__(model_factory, param_values, scoring_func, train_loader, validation_loader)
        self.param_domain = iter(RandomParametersDomain(param_values))
        self.num_iterations = num_iterations

    def get_tuned_model(self):
        best_score = None
        best_model = None
        best_params = None
        for i in range(0, self.num_iterations):
            hyper_params = next(self.param_domain)
            model = self.model_factory(**hyper_params)
            network_scorer = DenseNetworkScorer(model, hyper_params, self.scoring_func, self.train_loader,
                                                self.validation_loader)
            print('Scoring Network with params ', hyper_params)
            score = network_scorer.score()
            print('Network Score: ', score)
            if best_score is None or self.scoring_func.is_better(score, best_score):
                best_score = score
                best_model = model
                best_params = hyper_params
        return best_model, best_score, best_params
