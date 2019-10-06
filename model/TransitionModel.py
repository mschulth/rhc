import numpy as np
from abc import ABC, abstractmethod


class TransitionModel(ABC):

    def __init__(self, x_dim=None, y_dim=None):
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.x = np.empty((0, x_dim))
        self.y = np.empty((0, y_dim))

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def predict(self, x, ret_var=False):
        pass

    @abstractmethod
    def add_data(self, x, y, refine_model=True, opt_hyperparams=True):
        pass
