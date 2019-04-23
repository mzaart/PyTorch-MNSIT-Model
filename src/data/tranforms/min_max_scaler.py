import torch
from .in_place_tranformer import InPlaceTransformer


class MinMaxScaler(InPlaceTransformer):

    def __init__(self, num_features, sample_min, sample_max, desired_min=None, desired_max=None):
        super().__init__()
        self.num_items = num_features
        self.sample_min = sample_min
        self.sample_max = sample_max
        self.desired_min = desired_min or self._get_scalar_tensor(0, num_features)
        self.desired_max = desired_max or self._get_scalar_tensor(1, num_features)

    @staticmethod
    def from_features(features, desired_min=None, desired_max=None):
        """
        A utility factory method for constructing a MinMaxScalar from a 2d array of features.
        This method implicitly calculates the minimum and maximum element of each sample.
        """
        sample_min = [col.min() for col in features.T]
        sample_max = [col.max() for col in features.T]
        return MinMaxScaler(features.shape[1], sample_min, sample_max, desired_min, desired_max)

    def transform_x(self, x):
        transformed = torch.zeros_like(x)
        for i in range(0, x.shape[0]):
            transformed[i] = self._scale(x[i], i)
        return transformed

    def _scale(self, xi, i):
        xi_std = (xi - self.sample_min[i]) / (self.sample_max[i] - self.sample_min[i])
        xi_scaled = xi_std * (self.desired_max[i] - self.desired_min[i]) + self.desired_min[i]
        return xi_scaled

    def _get_scalar_tensor(self, scalar, *dimensions):
        tensor = torch.Tensor(*dimensions)
        tensor.fill_(scalar)
        return tensor
