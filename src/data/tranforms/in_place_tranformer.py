import abc
import torch


class InPlaceTransformer(abc.ABC):

    """
    This transformer is used to record the transformations in place in the original data set.
    This is class is useful when:
        - The transformations are costly to compute
        - AND items of the data set are read more than once

    Note that the transformation should not change the tensor dimensions.
    """

    _TRANSFORMED_FLAG = '_transformed'

    def transform_x(self, x):
        return x

    def transform_y(self, y):
        return y

    def __call__(self, item):
        if self._TRANSFORMED_FLAG not in item:
            x, y = item['x'], item['y']
            if x.shape:
                x[range(0, x.shape[0])] = self.transform_x(x)
            else:
                x.fill_(self.transform_x(x))

            if y.shape:
                y[range(0, y.shape[0])] = self.transform_y(y)
            else:
                y.fill_(self.transform_y(y))
            item[self._TRANSFORMED_FLAG] = True
        return item
