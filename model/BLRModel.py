import numpy as np
import casadi.casadi as cas
from abc import abstractmethod, ABC
from model.TransitionModel import TransitionModel


class BLRModel(TransitionModel, ABC):

    def __init__(self, x_dim, y_dim, n_features, beta=1.):
        super().__init__(x_dim, y_dim)

        self.mean = None
        self.sinv = None
        self.s = None
        self.beta = beta

        self.n_feats = n_features
        self.opt_maxsteps = 150

        self.mean0 = np.zeros((self.n_feats, self.y_dim))
        self.sinv0 = np.eye(self.n_feats) * 1e-8
        self.s0 = np.linalg.inv(self.sinv0)

        self._fcas = {}
        self._log_det_f = {}

    def reset(self):
        self.mean = self.mean0
        self.sinv = self.sinv0
        self.s = self.s0
        self._fcas = {}

    def retrain(self, opt_hyperparams=False):
        """
        Retrains the model
        :param opt_hyperparams: Whether to optimize the hyperparameters
        """
        if opt_hyperparams:
            self.opt_hyperparams()
        self.reset()
        self.update_param_dist(self.x, self.y)

        self._fcas = {}

    def update_param_dist(self, x, y):
        """
        Updates the parameter distribution with given inputs x and targets
        :param x: inputs
        :param y: targets
        """
        sinv0 = self.sinv
        mean0 = self.mean

        phi_x = self.phi(x)
        sinv_new = sinv0 + self.beta * np.dot(phi_x.T, phi_x)
        mean_new = np.linalg.solve(sinv_new, np.dot(sinv0, mean0) + self.beta * np.dot(phi_x.T, y))

        self.mean = mean_new
        self.sinv = sinv_new
        self.s = np.linalg.inv(sinv_new)

    def predict(self, x, ret_var=False):
        """
        Predicts the target values for inputs x
        :param x: the inputs
        :param ret_var: Whether to return the predictive variance
        :return: target value, predictive variance if ret_var=True
        """
        phi = self.phi(x)

        mu = np.dot(self.mean.T, phi.T).T
        if ret_var:
            sigma = 1 / self.beta + np.sum(phi * (self.s @ phi.T).T, axis=-1)
            return [mu, sigma]
        else:
            return mu

    def predict_casf(self, ret_var=False):
        """
        Returns a casadi function for making predictions
        :param ret_var: Whether the casadi function should also return the predictive variance
        :return: A casadi function that takes input and
        """
        if ret_var not in self._fcas:
            x = cas.MX.sym("x", 1, self.x_dim)

            phi = self.phi_cas(x)
            mu = cas.mtimes(self.mean.T, phi.T).T
            if ret_var:
                sigma = 1 / self.beta + cas.sum2(phi * cas.mtimes(self.s, phi.T).T)
                res = [mu, sigma]
            else:
                res = [mu]
            self._fcas[ret_var] = cas.Function("f_mu", [x], res)

        return self._fcas[ret_var]

    @abstractmethod
    def phi(self, x):
        """
        Computes the feature representation for x
        :param x: the input x
        :return: the feature representation
        """
        pass

    @abstractmethod
    def phi_cas(self, x):
        """
        Computes the feature representation for casadi variable x
        :param x: The input casadi variable
        :return: the feature representation as casadi variable
        """
        pass

    def pred_ent_cas(self, x):
        """
        Computes the predicted entropy of the model parameter distribution if
        refining the model with observations x.
        :param x: The observations to refine the model with
        :return: the entropy as casadi variable
        """
        phi_x = self.phi_cas(x)
        sinv_new = self.sinv + self.beta * cas.mtimes(phi_x.T, phi_x)

        log_det_f = self.log_det_cas(sinv_new.shape[0])
        s = -log_det_f(sinv_new)

        k = self.n_feats
        ent = k / 2 + k / 2 * cas.log(2 * cas.pi) + s / 2

        return ent

    def log_det_cas(self, size):
        """
        Returns a casadi function to compute the log determinant of a matrix. If the function with the specific
        size was already requested, the cached function will be returned for efficiency
        :param size: the dimension of the square matrix
        :return: a casadi function for computing the log determinant
        """
        if size in self._log_det_f:
            return self._log_det_f[size]
        else:
            S = cas.SX.sym("s", size, size)
            f = cas.Function('log_det', [S], [cas.trace(cas.log(cas.qr(S)[1]))]).expand()
            self._log_det_f[size] = f
            return f

    def add_data(self, x, y, refine_model=True, opt_hyperparams=True):
        """
        Adds data to the model and refines the parameter distribution if desired
        :param x: The inputs of the data to add
        :param y: The targets of the data to add
        :param refine_model: Whether to refine the model using the new data
        :param opt_hyperparams: Whether to optimize the hyperparameters of the model.
                                Is only considered if refine_model=True.
        """
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))

        if refine_model:
            if opt_hyperparams:
                self.retrain(opt_hyperparams=True)
            else:
                self.update_param_dist(x, y)

    def param_entropy(self):
        """
        Calculates the entropy of the model parameters
        :return: the entropy
        """
        sinv = self.sinv
        k = self.n_feats

        (sign, ld) = np.linalg.slogdet(sinv)
        logdets = -ld

        ent = np.log(2 * np.pi * np.e) * k / 2 + logdets/2
        return ent

    @abstractmethod
    def opt_hyperparams(self):
        """
        Optimizes the hyperparameters of the model by maximizing the likelihood
        :return: the new likelihood achieved by the optimal parameters
        """
        pass
