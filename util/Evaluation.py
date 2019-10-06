import numpy as np
from util.DatasetHandler import DatasetHandler
from abc import ABC, abstractmethod
import logging
from util.Planning import Planning


class Evaluation(ABC):

    def __init__(self, model, env=None, logger=None):
        self.model = model
        self.env = env

        if logger is None:
            self.logger = logging.getLogger("exploration")
        else:
            self.logger = logger

    @abstractmethod
    def eval(self):
        pass


class EvaluationCollection(Evaluation):

    def __init__(self, *evaluators):
        self.evaluators = evaluators

    def eval(self):
        res = {}
        for e in self.evaluators:
            eres = e.eval()
            res = {**res, **eres}
        return res


class TaskEvaluation(Evaluation):

    def __init__(self, model, env=None, logger=None):
        super().__init__(model, env, logger)

        self.max_iter = 1500
        self.seed = 0
        self.state_cost_start = 0

    def eval(self):
        if self.seed is not None:
            rs = np.random.get_state()
            np.random.seed(self.seed)

        # do planning
        pl = Planning(self.env, self.model)
        pl.opt_maxiter = self.max_iter
        cost = self.env.cost_cas(state_cost_start=self.state_cost_start)
        t = self.env.task_t
        x0 = self.env.obs_x0
        opt_a, opt_x, opt_varx, cost = pl.plan_multiple_shooting(t, cost, x0, self.env.x_bounds)

        # execute in environment
        s, _ = self.env.run_actions(opt_a)

        # calculate cost
        cost = self.env.cost(s, opt_a)

        res = {
            'task_cost': cost
        }

        # roll back random state
        if self.seed is not None:
            np.random.set_state(rs)

        self.logger.info("Task evaluation done: {}".format(res))

        return res


class DatasetEvaluation(Evaluation):

    def __init__(self, model, env=None, logger=None):
        super().__init__(model, env, logger)

        if env is not None:
            self.test_set = DatasetHandler.get_data_set_test(env)
        else:
            self.test_set = None

    def eval(self):
        """
        Evaluates the model in self.model on the test data set stored in self.test_set
        The test set may be of ndim = 2 (just transitions: [N, X+Y]))
        or of ndim = 3 for trajectories ([N, T, X+Y]).
        :return: A dictionary of evaluation results
        """
        if self.test_set is None:
            raise AttributeError("A test set must be set")

        test_set = self.test_set
        results = {}

        tdim = np.ndim(test_set)
        if tdim == 3:
            # If the test set consists of trajectories, compute T-step error and likelihood
            s_dim = self.model.y_dim
            T = test_set.shape[1]

            # simulate model
            x_true = test_set[:, :, :self.model.x_dim]
            x_model = np.copy(x_true)
            x_model[:, 1:, :s_dim] = np.nan
            y_true = test_set[:, :, self.model.x_dim:]
            y_pred = np.copy(y_true) * np.nan
            y_pred_var = np.empty((y_true.shape[0], y_true.shape[1]))
            for t in range(T):
                xt = x_model[:, t, :self.model.x_dim]
                pred, var = self.model.predict(xt, ret_var=True)
                y_pred[:, t, :] = pred
                y_pred_var[:, t] = var
                if t + 1 < T:
                    x_model[:, t + 1, :s_dim] = pred + xt[:, :s_dim]

            # accumulate predictions for n_step prediction
            y_n_step_true = np.sum(y_true, axis=1)
            y_n_step_pred = np.sum(y_pred, axis=1)

            # calculate n-step error and llh
            n_step_rmse = np.sqrt(np.mean(np.square(y_n_step_true - y_n_step_pred)))
            n_step_llh = self.llh_pred(y_pred, y_pred_var, y_true)

            results['n_step_rmse'] = n_step_rmse
            results['n_step_llh'] = n_step_llh

            # take only first time step to calculate 1-step error
            test_set_os = self.test_set[:, 0, :]
        elif tdim == 2:
            test_set_os = self.test_set
        else:
            raise AttributeError("Test set must have two or three dimensions.")

        # compute one-step predictions
        x = test_set_os[:, :self.model.x_dim]
        y_true = test_set_os[:, -self.model.y_dim:]
        y_pred, y_pred_var = self.model.predict(x, ret_var=True)

        diff = y_pred - y_true
        err = np.sqrt(np.mean(np.square(diff), axis=0))
        llh = self.llh_pred(y_pred, y_pred_var, y_true)
        err_mn = np.mean(err)

        results['rmse_dimwise'] = err
        results['rmse'] = err_mn
        results['llh'] = llh

        self.logger.info("Dataset evaluation done: {}".format(results))
        return results

    def llh_pred(self, pred_mean, pred_var, true):
        """
        Computes the Gaussian likelihood of given data
        :param pred_mean: the predictive Gaussian mean
        :param pred_var: the predictive Gaussian variance
        :param true: the true values
        :return: the likelihood of the true data
        """

        k = true.shape[-1]
        ndim = np.ndim(pred_mean)

        y_diff = true - pred_mean
        llht = np.sum(y_diff ** 2, axis=-1) / pred_var

        logdetsigmas = np.log(pred_var)
        if ndim == 3:
            n = llht.shape[1]
            llht2 = np.sum(llht, axis=-1)
            logdetsigmas2 = np.sum(logdetsigmas, axis=-1)
        else:
            n = 1
            llht2 = llht
            logdetsigmas2 = logdetsigmas

        llh = -(np.mean(logdetsigmas2) + np.mean(llht2) + k * n * np.log(2 * np.pi)) / 2

        return llh
