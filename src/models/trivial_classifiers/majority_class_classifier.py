from torch import nn


class MajorityClassClassifier(nn.Module):

    def __init__(self, majority_class):
        super().__init__()
        self.majority_class = majority_class

    def forward(self, x):
        return self.majority_class
