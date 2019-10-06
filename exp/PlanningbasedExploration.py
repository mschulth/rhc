import casadi.casadi as cas
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from util.Planning import Planning
from exp.Exploration import Exploration, MaxEpisodesReachedException


class PlanningbasedExploration(Exploration, ABC):

    def __init__(self, env, model, name="rhc", evaluation=None):
        super().__init__(env, model, name, evaluation)
        self.planning = Planning(env, model)

    @abstractmethod
    def plan_exploration_trajectory(self, t, x0, x_init=None):
        """
        Plans a trajectory for optimization
        :param t: the time horizon to plan for
        :param x0: the initial state
        :param x_init: an initialization of the trajectory
        :return: the optimized actions, states, transition variances, and reached cost
        """
        return None, None, None, None

    def run(self):
        """
        Executes the system identification
        """
        self._init()

        for i in range(self.n_episodes):
            self.logger.info("Exploration episode {}".format(i+1))

            # plan trajectory
            u, s, varx, cost = self.plan_exploration_trajectory(t=self.env.task_t, x0=self.env.obs_x0)

            # execute on system
            u_true, s_true, fx, fy = self.env.run_actions_as_feats(u)

            if self.save_plots:
                ax1 = plt.figure()
                self.env.plot(s[1:], u, varx=varx, ax=ax1, postfix="_plan")
                self.env.plot(s_true, u, varx=None, ax=ax1, postfix="_true")
                plt.title("{} - optimisation episode {}".format(self.name, i+1))
                plt.legend()
                plt.savefig(os.path.join(self.plot_path, 'plt_plan_{}.png'.format(self.episode)))
                plt.close()

            try:
                self._end_episode(fx, fy, s_true, u_true)
            except MaxEpisodesReachedException:
                pass


class MaxTrajectoryEntropyExploration(PlanningbasedExploration):

    def __init__(self, env, model, name="rhc_us", evaluation=None):
        super().__init__(env, model, name, evaluation)
        self.logger.info("RHC: Uncertainty Sampling Exploration selected.")

    def plan_exploration_trajectory(self, t, x0, x_init=None):
        # create cost function
        s = cas.MX.sym("s", t + 1, self.env.o_dim)
        a = cas.MX.sym("a", t, self.env.a_dim)
        v = cas.MX.sym("v", t, 1)
        cost = -cas.sum1(v)
        fcost = cas.Function("reward", [s, a, v], [cost])

        # planning
        u_opt, s_opt, varx_opt, cost = self.planning.plan_multiple_shooting(t, fcost, x0, x_init=x_init)

        return u_opt, s_opt, varx_opt, cost


class MinModelEntropyExploration(PlanningbasedExploration):

    def __init__(self, env, model, name="rhc_mvr", evaluation=None):
        super().__init__(env, model, name, evaluation)
        self.logger.info("RHC: Maximal Variance Reduction Exploration selected.")

    def plan_exploration_trajectory(self, t, x0, x_init=None):
        # define objective
        s = cas.MX.sym("s", t + 1, self.env.o_dim)
        a = cas.MX.sym("a", t, self.env.a_dim)
        v = cas.MX.sym("v", t, 1)
        x = cas.horzcat(s[:-1, :], a)

        cost = self.model.pred_ent_cas(x)
        fcost = cas.Function("reward", [s, a, v], [cost])

        # planning
        u_opt, s_opt, varx_opt, cost = self.planning.plan_multiple_shooting(t, fcost, x0, x_init=x_init)

        return u_opt, s_opt, varx_opt, cost
