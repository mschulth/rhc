import numpy as np
from abc import ABC, abstractmethod
import logging
import os
import matplotlib.pyplot as plt
import pickle


class Exploration(ABC):

    def __init__(self, env, model, name="", evaluation=None, logger=None):
        self.env = env
        self.model = model
        self.name = name
        self.evaluation = evaluation

        self.n_episodes = 20       # number of episodes of exploration
        self.horizon = env.task_t  # time horizon of one episode
        self.seed = None
        self.save_plots = True
        self.filter_feats_to_bounds = True

        if logger is None:
            self.logger = logging.getLogger("exploration")
        else:
            self.logger = logger

        self.episode = 1
        self.evals = []
        self.results_path = "results"
        self.plot_path = os.path.join(self.results_path, self.name)

    @abstractmethod
    def run(self):
        """
        Launches the exploration process for self.n_episodes
        """
        pass

    def _init(self):
        self.logger.info("Initialize exploration.")
        self.episode = 1
        self.evals = [self.evaluation.eval()]
        self.env.reset()

        if self.save_plots:
            os.makedirs(self.plot_path, exist_ok=True)

        if self.seed is not None:
            np.random.seed(self.seed)

    def _end_episode(self, feat_x, feat_y, s=None, a=None):
        self.logger.info("Exploration episode no. {} terminated.".format(self.episode))

        # add observations to model
        if self.filter_feats_to_bounds:
            feat_x, feat_y = self.filter_feats(feat_x, feat_y)
        self.model.add_data(feat_x, feat_y, refine_model=True, opt_hyperparams=True)

        # save and evaluate
        self.evals += [self.evaluation.eval()]

        if self.save_plots and s is not None and a is not None:
            ax1 = plt.figure()
            self.env.plot(s, a, ax=ax1)
            plt.title("{} - episode {}".format(self.name, self.episode))
            plt.legend()
            plt.savefig(os.path.join(self.plot_path, 'plt_{}.png'.format(self.episode)))
            plt.close()

        if self.episode >= self.n_episodes:
            self.logger.info("Max num of episodes reached ({}). Terminate exploration.".format(self.episode))
            raise MaxEpisodesReachedException()

        # reset env
        self.episode += 1
        self.env.reset()

    def filter_feats(self, x, y):
        num_features = np.shape(x)[0]
        if num_features == 0:
            return x, y

        next_states = x[:, :np.shape(y)[1]] + y
        valid = self.env.state_valid(next_states)
        limit = np.argmax(np.logical_not(valid))

        if limit == 0:
            if valid[limit]:
                limit = num_features

        self.logger.info("Filtered {} of {} features".format(num_features - limit, num_features))

        return x[:limit, :], y[:limit, :]

    def save_results(self):
        os.makedirs(self.results_path, exist_ok=True)
        file = os.path.join(self.results_path, "res_"+self.name+".pkl")
        self.logger.info("Saving evaluation results to '{}'.".format(file))
        with open(file, 'wb') as f:
            pickle.dump(self.evals, f)


class MaxEpisodesReachedException(Exception):
    """
    Exception to raise when the maximum number of episodes has been reached to end exploration.
    Used for example for model_free reinforcement learning methods where no
    """
    pass
