from torch import nn


class DenseNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, activation_factory, out_func, h_dim=(), drop_out=None):
        """
        Represents a dense neural network.

        :param in_dim: The dimensions of the model input

        :param out_dim: The dimensions of the model output

        :param activation_factory: A function that returns a new instance of an activation function each time it
        is called. The returned activation will be used in the hidden layers.

        :param out_func: The activation function to be used by the out put layer.

        :param h_dim: A list containing the number of neurons in each hidden layer.

        :param drop_out: A list containing probabilities that are applied to hidden layers.
        dropout[i] represents the dropout probability for the ith layer. You can set dropout to None if you want
        to disable dropouts for the network or dropout[i] to None to disable dropout for the ith layer.
        If dropout is a float, the same dropout will be applied to all hidden layers.
        """
        super().__init__()
        self.in_dim = in_dim
        self.activations = (activation_factory, out_func)
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.drop_out = drop_out

        # calc some network info
        self.num_neural_layers = len(h_dim) + 1
        self.num_activation_layers = self.num_neural_layers
        self.num_drop_out_layers = self.num_neural_layers - 1 if drop_out else 0
        self.num_layers = self.num_neural_layers + self.num_activation_layers + self.num_drop_out_layers

        self._build_network()

    def _build_network(self):
        # construct iterators
        layer_names = iter(['layer_{}'.format(i) for i in range(0, self.num_layers)])
        if self.drop_out is None:
            drop_out = None
        elif isinstance(self.drop_out, float):
            drop_out = iter([self.drop_out for _ in range(0, self.num_layers)])
        elif isinstance(self.drop_out, list):
            drop_out = iter([p if p else 0 for p in self.drop_out])
        else:
            raise ValueError('Invalid type {} for dropout'.format(type(self.drop_out)))

        # add input layer
        input_layer = nn.Linear(
            self.in_dim,
            self.h_dim[0] if self.h_dim else self.out_dim
        )
        self.add_module(next(layer_names), input_layer)
        if drop_out:
            self.add_module(next(layer_names), nn.Dropout(next(drop_out)))
        self.add_module(next(layer_names), self.activations[0]())

        # add other layers
        for i, h_dim in enumerate(self.h_dim):
            linear_layer = nn.Linear(
                h_dim,
                self.h_dim[i+1] if i < len(self.h_dim) - 1 else self.out_dim
            )
            self.add_module(next(layer_names), linear_layer)
            if i == len(self.h_dim) - 1:
                self.add_module(next(layer_names), self.activations[1])
            else:
                if drop_out:
                    self.add_module(next(layer_names), nn.Dropout(next(drop_out)))
                self.add_module(next(layer_names), self.activations[0]())

    def forward(self, x):
        for module in self.children():
            x = module.forward(x)
        return x
