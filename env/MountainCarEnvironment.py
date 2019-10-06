import numpy as np
import gym
from env.Environment import Environment


class MountainCarEnvironment(Environment):

    def __init__(self):
        super().__init__()

        self.env = gym.make('MountainCarContinuous-v0')
        self.env.env.power = .001
        self.env.env.max_speed = np.inf
        self.env.env.low_state[1] = -self.env.env.max_speed
        self.env.env.high_state[1] = self.env.env.max_speed

        self.x0 = np.array([-0.5236, 0])
        self.s_dim = 2
        self.o_dim = 2
        self.a_dim = 1
        self.a_lim = np.vstack((self.env.action_space.low, self.env.action_space.high)).T
        self.s_lim = np.array([[-1.15, 0.5], [-0.065, 0.065]])
        self.o_lim = np.array([[-1.15, 0.55], [-np.inf, np.inf]])
        self.o_labels = ["s", "v"]
        self.name = "mc"

        self.task_goal = np.array([.5, 0.])
        self.task_cost_weights = np.array([10, 0, .01])
        self.task_num_rff_feats = 20
        self.task_t = 130
        self.task_state_cost_start = -1

    def step(self, u):
        self.env.step(u)
        return self.obs_state

    def reset(self):
        self.env.reset()
        self.state = self.x0
        return self.obs_state

    @property
    def state(self):
        state = self.env.env.state
        return np.atleast_2d(state)

    @state.setter
    def state(self, s):
        s = np.ravel(s)
        self.env.env.state = s

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
