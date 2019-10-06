import numpy as np
import casadi as cas
import scipy.optimize as sopt
from model.BLRModel import BLRModel


class RFFModel(BLRModel):

    def __init__(self, x_dim, y_dim, n_features, v=1., beta=1.):
        """
        :param x_dim:
        :param y_dim:
        :param n_features: The number of features used for the RFF model
        :param v: The initial v parameter for the RFF features
        :param beta: The beta parameter for the linear regression
        """
        super().__init__(x_dim=x_dim, y_dim=y_dim, n_features=n_features, beta=beta)

        self.W = None
        self.psi = None

        # if saclar
        if np.isscalar(v) or np.size(v) == 1:
            assert np.size(v) == 1
            self.v = np.repeat(v, self.x_dim)
        else:
            assert(np.ndim(v) == 1 and v.shape[0] == self.x_dim)
            self.v = v

        self.reset()

    def reset(self):
        super().reset()

        self.W = np.random.normal(0, 1, (self.n_feats, self.x_dim))
        self.psi = np.random.uniform(0, 2*np.pi, self.n_feats)

        # introduce bias features
        self.W[0, :] = 0
        self.psi[0] = 0

    def phi(self, x, v=None):
        """
        Evaluates the RBF model on observations
        :param x: 2d array of shape [N, D] holding the N observations (of dimension D each)
        :return: 2d array of size [N, M] holding the matrix phi
        """
        if v is None:
            v = self.v

        x = np.atleast_2d(x)
        assert(x.shape[1] == self.x_dim)

        y = np.sin((self.W / v[None, :]) @ x.T + self.psi[:, None])
        return y.T

    def phi_cas(self, x, v=None):
        if v is None:
            v = self.v[None, :]
        else:
            v = cas.repmat(v, self.W.shape[0], 1)
        assert (x.shape[1] == self.x_dim)

        y = cas.sin(cas.mtimes(self.W / v, x.T) + self.psi)
        return y.T

    def opt_llh_NM(self):
        """
        Maximizes the likelihood w.r.t. the hyperparameters v using gradient-free Nelder Mead method.
        :return: the optimal v and the achieved likelihood
        """
        -self.llh(self.v)
        nllh = lambda v: -self.llh(v)
        res = sopt.minimize(nllh, x0=self.v, method='Nelder-Mead')
        return res.x, res.fun

    def opt_llh_cas(self, v0=None):
        """
        Maximizes the likelihood w.r.t. the hyperparameters v using gradient-based ipopt method.
        :param v0: The initial value for v. If None, ones will be used.
        :return: the optimal v and the achieved likelihood
        """
        vshape = np.atleast_2d(self.v).shape
        v = cas.MX.sym("v", vshape[0], vshape[1])
        f_cas = self.nllh_casf(grad=False, hess=False)
        obj = f_cas(v)[0]

        nlp = {'x': v, 'f': obj}
        solver = cas.nlpsol("solver", "ipopt", nlp, {'ipopt.max_iter': self.opt_maxsteps})

        if v0 is None:
            v0 = np.ones(np.shape(self.v))

        # solve nlp
        sol = solver(x0=v0, lbx=0, lbg=0, ubg=0)
        v_opt = np.squeeze(np.array(sol['x']))
        return v_opt, float(sol['f'])

    def opt_hyperparams(self):
        #x, nllh = self.opt_llh_cas(None)
        x, nllh = self.opt_llh_NM()
        self.v = x
        return nllh

    def llh(self, v=None, x_eval=None, y_eval=None):
        """
        Computes the likelihood (without constant terms) of the data dependent on parameters v without const term
        :param v: the hyperparameter v. If none, ones will be used.
        :param x_eval: the inputs of the data to compute the likelihood of. If none, the model's dataset will be used.
        :param y_eval: the targets of the data to compute the likelihood of. If none, the model's dataset will be used.
        :return: the likelihood
        """
        assert ((x_eval is None) == (y_eval is None))

        if x_eval is None:
            x_eval = self.x
            y_eval = self.y

        if v is not None:
            phi_x = self.phi(self.x, v)
            sinv = self.sinv0 + self.beta * phi_x.T @ phi_x
            mean = np.linalg.solve(sinv, self.sinv0 @ self.mean0 + self.beta * (phi_x.T @ self.y))
            phi_x_eval = phi_x
        else:
            phi_x_eval = self.phi(x_eval)
            sinv = self.sinv
            mean = self.mean

        y_pred = (mean.T @ phi_x_eval.T).T
        sigma = 1 / self.beta + np.sum(phi_x_eval * (np.linalg.solve(sinv, phi_x_eval.T)).T, axis=1)

        n = x_eval.shape[0]
        y_diff = y_eval - y_pred
        llht = np.sum(y_diff ** 2, axis=1) / sigma

        llh = -np.sum(llht)
        # llh = -np.sum(llht) - np.sum(np.log(sigma))/2 - n*np.log(2*np.pi)/2  # correct one with constant terms
        return llh

    def nllh_casf(self, grad=True, hess=False):
        """
        Creates a casadi function computing the negative log likelihood (without const terms) of the data
        dependent on hyperparameters v.
        :param grad: Whether the function should compute the gradient, too
        :param hess: Whether the function should compute the hessian, too
        :return: A casadi function taking a value of v as input and returning the neg log likelihood, gradient,
                 and hessian is desired
        """
        vshape = np.atleast_2d(self.v).shape
        v = cas.MX.sym("v", vshape[0], vshape[1])
        phi_x = self.phi_cas(self.x, v)

        sinv = self.sinv0 + self.beta * cas.mtimes(phi_x.T, phi_x)
        mean = cas.solve(sinv, cas.mtimes(self.sinv0, self.mean0) + self.beta * cas.mtimes(phi_x.T, self.y))

        y_pred = cas.mtimes(mean.T, phi_x.T).T
        sigma = 1 / self.beta + cas.sum2(phi_x * cas.solve(sinv, phi_x.T).T)

        y_true = self.y
        y_diff = y_true - y_pred
        llht = cas.sum2(y_diff * y_diff) / sigma
        llh = cas.sum1(llht)

        if hess:
            H, llh_grad = cas.hessian(llh, v)
        elif grad:
            llh_grad = cas.gradient(llh, v)

        res = [llh]
        if grad:
            res += [llh_grad]
        if hess:
            res += [H]

        f = cas.Function("f_mu", [v], res)
        return f
