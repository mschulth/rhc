import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import casadi as cas
from gym.spaces import Box


class Environment(ABC):
    """ This class represents an environment that can be controlled by an agent. """

    def __init__(self):
        self.a_dim = None  # action dimensionality
        self.s_dim = None  # state dimensionality (the state is only internal)
        self.o_dim = None  # observation dimensionality (may be different from the internal state)

        self.a_lim = None     # action limits
        self.s_lim = None     # limits of the state space
        self.o_lim = None     # limits of the observation space, used for normalization

        self.o_labels = None  # labels for observation dimensions, used for plotting

        self.x0 = None        # the initial state

        self.task_cost_weights = None   # cost weights for solving the task
        self.task_goal = None           # the goal (in observation space) of the task
        self.task_num_rff_feats = None  # the number of rff used by default to learn a model
        self.task_t = None              # the time horizon for solving a task
        self.task_state_cost_start = 0  # time step from where on to count cost of the goal distance

        # The following variables will be set automatically (lazy variables)
        self._x0_obs = None
        self._x_bounds = None
        self._action_space = None
        self._observation_space = None
        self._state = None

    def reset(self):
        """
        Resets the environment
        :return: the observation of the initial state
        """
        self.state = self.x0
        return self.obs_state

    @abstractmethod
    def step(self, u):
        """
        Executes a step in the environment
        :param u: The action to execute
        :return: The observation of the state after executing the action
        """
        pass

    def obs(self, s):
        """
        Returns the observation of a state
        :param s: the state
        :return: the observation
        """
        return s

    def render(self):
        """
        Rendering functionality (if supported by the environment)
        :return:
        """
        pass

    def close(self):
        pass

    # --------------------------------------------------------------
    # ----------------------------------------------------------
    # The methods below usually don't need to be overridden.

    def run_actions_as_feats(self, actions, reset=True):
        s, u = self.run_actions(actions, reset=reset)
        x, y = self.features(s, actions)
        return u, s, x, y

    def features(self, s, us):
        """
        Converts states and actions into features for learning a model
        :param s: states
        :param us: actions
        :return: x, y features
        """
        us = np.atleast_2d(us)
        s = np.atleast_2d(s)
        x = np.hstack((s[:-1, :], us))
        y = s[1:, :] - s[:-1, :]
        return x, y

    def obs_x(self, x):
        x = np.atleast_2d(x)
        xobs = np.hstack((self.obs(x[:, :self.s_dim]), x[:, self.s_dim:]))
        return xobs

    @property
    def x_bounds(self):
        if self._x_bounds is None:
            if self.o_lim is None:
                return None
            else:
                self._x_bounds = np.vstack((self.o_lim, self.a_lim))
        return self._x_bounds

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = Box(self.a_lim[:, 0], self.a_lim[:, 1], dtype=np.float64)

        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            self._observation_space = Box(-np.inf, np.inf, (self.o_dim,), dtype=np.float64)

        return self._observation_space

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s

    @property
    def obs_state(self):
        return self.obs(self.state)

    @property
    def obs_x0(self):
        if self._x0_obs is None:
            self._x0_obs = self.obs(self.x0)

        return self._x0_obs

    @property
    def x_dim(self):
        return self.o_dim + self.a_dim

    @property
    def x_lim(self):
        return np.vstack((self.o_lim, self.a_lim))

    def execute_rand(self, num_actions, reset=False):
        """
        Executes uniformly random sampled actions
        :param num_actions: the number of actions to execute
        :param reset: whether to reset the environment before executing actions
        :return: states, features_x, features_y, actions
        """
        rand = np.random.random_sample((num_actions, self.a_dim))
        upper = self.a_lim[:, 1]
        lower = self.a_lim[:, 0]
        dist_uplow = upper - lower
        a = rand * dist_uplow + lower
        _, s, x, y = self.run_actions_as_feats(a, reset=reset)
        return a, s, x, y

    def clip_actions(self, u):
        return np.clip(u, self.a_lim[:, 0], self.a_lim[:, 1])

    def run_actions(self, u, repeat_infinite=False, plot=False, render=False, vr=None, reset=True):
        """
        New method for simulating actions that also supports rendering after each step and multiple sequences
        :param u: The actions to execute
        :param repeat_infinite: Whether to repeat rendering infinitely often (only if render is True)
        :param plot: Whether to plot the simulation results
        :param render: Whether to render after each execution step
        :param vr: Video recorder to record nice videos for my presentation
        :param reset: whether the environment should be resetted before executing the actions
        :return: The visited states
        """
        if repeat_infinite and not render:
            repeat_infinite = False

        if u.ndim == 1 or (np.shape(u)[-1] != 1 and self.a_dim == 1):
            u = np.expand_dims(u, axis=-1)

        if u.ndim == 2:
            u = np.expand_dims(u, axis=0)

        for ex in range(u.shape[0]):
            u[ex] = self.clip_actions(u[ex])
            if reset:
                self.reset()
            if render:
                for _ in range(5):
                    self.render()
                    if vr is not None:
                        vr.capture_frame()

            s = self.obs_state

            for i in range(u[ex].shape[0]):
                a = u[ex][i, :]
                self.step(a)

                if render:
                    self.render()
                    if vr is not None:
                        vr.capture_frame()

                s = np.vstack((s, self.obs_state))

            if plot:
                uflat = np.reshape(u, (-1, u.shape[-1]))
                self.plot(s, uflat, "Simulated")

        if repeat_infinite:
            return self.run_actions(u, repeat_infinite=repeat_infinite, plot=False)
        else:
            if render:
                self.close()
            return s, u

    def plot(self, o, a, title=None, varx=None, ax=None, postfix=""):
        """
        Plots a trajectory.
        For labels of the plots, self.o_labels is used if set.
        :param o: Sequence of observations
        :param a: sequence of actions
        :param title: Title of the plot
        :param varx: State variances for plotting
        :param ax: plot axis for plotting into an already-exisiting figure
        :param postfix: postfix for the plot labels.
        :return:
        """

        if ax is None:
            axr = plt.figure()
            if title is not None:
                plt.title(title)
        else:
            axr = ax

        if a is not None:
            a = np.squeeze(a)
            if a.ndim == 1:
                a = a[:, None]

            for i in range(self.a_dim):
                ai = a[:, i]
                plt.plot(ai, label="a{}{}".format(i, postfix))

        if o.ndim == 1:
            o = o[:, None]
        for i in range(self.o_dim):
            if self.o_labels is not None:
                lbl = self.o_labels[i]
            else:
                lbl = "x{}".format(i)
            self.var_plt(o[:, i], label=lbl + postfix, var=varx)

        if ax is None:
            plt.legend()
            plt.show()

        return axr

    def var_plt(self, x, y=None, label=None, var=None):
        """
        Creates a line plot with variances
        :param x: the x values for plotting
        :param y: the y values for plotting
        :param label: the label for the lines
        :param var: the variances to plot
        :return: the line of the plot
        """
        if y is not None:
            line = plt.plot(x, y, label=label)
        else:
            line = plt.plot(x, label=label)

        if var is not None:
            #var = np.sqrt(np.squeeze(var))
            var = np.log(var + 1e-5)[:, 0]/2
            col = line[0]._color
            plt.fill_between(np.arange(x.shape[0]), x - var, x + var,
                             color=col, alpha=0.2)

        return line

    def cost(self, s, a, state_cost_start=None, goal=None, weights=None):
        """
        Calculates the cost for a trajectory of the environment.
        By default, a squared error from the goal and squared action penalty is used.
        :param s: the states of the trajectory
        :param a: the actions of the trajectory
        :param state_cost_start: the time step when the state cost is considered fist
        :param goal: the goal
        :param weights:
        :return:
        """
        if weights is None:
            weights = self.task_cost_weights

        if goal is None:
            goal = self.task_goal

        if state_cost_start is None:
            state_cost_start = self.task_state_cost_start

        assert(weights.size == self.x_dim)
        assert(goal.size == self.o_dim)

        nones = np.argwhere(np.isnan(goal.ravel()))
        goal.ravel()[nones] = 0
        weights[nones] = 0

        goal = np.reshape(goal, (1, -1))

        dist_goal = s - goal
        a = np.atleast_2d(a)
        # expand a to the number of states
        a_ext = np.zeros((dist_goal.shape[0], a.shape[1]))
        a_ext[:a.shape[0], :] = a

        if state_cost_start is not None:
            if state_cost_start < 0:
                state_cost_start = self.task_t - state_cost_start

            if state_cost_start > 0:
                dist_goal[:state_cost_start, :] = 0

        vals = np.hstack((dist_goal, a_ext))
        cost = np.sum(weights * np.square(vals))
        return cost

    def cost_cas(self, goal=None, weights=None, var=True, state_cost_start=0):
        """
        Creates a casadi cost function
        :param goal: The goal. If none, the environment's default goal will be used
        :param weights: The weights for the state and action dimensions.
                        If none, the environment's default goal will be used
        :param var: Whether the input of the casadi function receives the variances
        :param state_cost_start: The time step when the state cost is considered fist
        :return:
        """
        if weights is None:
            weights = self.task_cost_weights

        if goal is None:
            goal = self.task_goal

        assert(weights.size == self.x_dim)
        assert(goal.size == self.o_dim)

        nones = np.argwhere(np.isnan(goal.ravel()))
        goal.ravel()[nones] = 0
        weights[nones] = 0

        goal = np.reshape(goal, (1, -1))
        xi = cas.MX.sym("xi", 1, self.o_dim)
        ui = cas.MX.sym("ui", 1, self.a_dim)
        fun_in = [xi, ui]
        if var:
            vxi = cas.MX.sym("vxi", 1, 1)
            fun_in += [vxi]

        dist_goal = xi - goal

        vals = cas.hcat((dist_goal, ui))

        if state_cost_start is None or state_cost_start == 0:
            state_cost_start = 0
        elif state_cost_start < 0:
            state_cost_start = self.task_t - state_cost_start

        # make return time dependent cost function
        def o(t):
            if t < state_cost_start - 1:
                pass
            else:
                return cas.Function("reward", fun_in, [cas.mtimes(vals, weights * vals.T)])

        #  o = lambda _: cas.Function("reward", fun_in, [cas.mtimes(vals, weights * vals.T)])

        return o

    def sim_states(self, s, render=True):
        """
        Simulates states in the environment
        :param s: The states to set
        :param render: Whether to render
        :return:
        """
        while True:
            for si in s:
                self.state = si
                if render:
                    self.render()

    def state_valid(self, s):
        """
        Checks if states are valid
        :param s: The states
        :return: 1d-array indicating for each state whether it is valid
        """
        s = np.atleast_2d(s)
        if self.o_lim is None:
            return np.repeat(True, s.shape[0])

        lower_valid = s > self.o_lim[:, 0] - 1e-5
        upper_valid = s < self.o_lim[:, 1] + 1e-5

        la = np.logical_and(lower_valid, upper_valid)
        valid = np.all(la, axis=1)

        return valid

    def eval(self, s, a):
        """
        Evaluates a trajectory for the environment
        :param s: The states of the trajectory
        :param a: The actions of the trajectory
        :return: A dictionary of evaluations
        """
        min_goal_dist = np.min(self.goal_dist(s))
        last_goal_dist = self.goal_dist(s)[-1]
        cost = self.cost(s, a)

        res = {
            'min_goal_dist': min_goal_dist,
            'last_goal_dist': last_goal_dist,
            'cost': cost
        }
        return res

    def goal_dist(self, s, goal=None, weights=None):
        """
        Computes the squared distance of states to a goal
        :param s: The states
        :param goal: The goal. If none, the environment's default goal will be used
        :param weights: The weights. If none, the environment's default weights will be used
        :return: A 1d-array indicating the distances of the states to the goals
        """
        if weights is None:
            weights = self.task_cost_weights
        if goal is None:
            goal = self.task_goal

        assert (weights.size == self.x_dim)
        assert (goal.size == self.o_dim)

        goal = np.reshape(goal, (1, -1))
        dist_goal = s - goal
        dists = np.sum(weights[:goal.shape[1]] * np.square(dist_goal), axis=1)
        return dists
