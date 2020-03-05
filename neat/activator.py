from scipy.special import expit
import numpy as np


class Activator:
    def __init__(self, name="sigmoid"):
        self.name = name
        self.funcs = {
            "sigmoid": self._sigmoid,
            "tanh": self._tanh
        }

    def get_func(self):
        return self.funcs[self.name]

    def compute_func(self, x):
        return self.funcs[self.name](x)

    @staticmethod
    def _sigmoid(x):
        return expit(x)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)
