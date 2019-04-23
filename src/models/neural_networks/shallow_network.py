from torch import nn
from .dense_network import DenseNetwork


class ShallowNetwork(DenseNetwork):

    def __init__(self):
        in_dim = 28 * 28
        out_dim = 10
        h_dim = [
            200,
            200,
        ]
        activation = lambda: nn.ReLU()
        out_func = nn.LogSoftmax(dim=1)
        super().__init__(in_dim, out_dim, activation, out_func, h_dim)
