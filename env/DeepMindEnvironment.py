import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from env.Environment import Environment
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import suite as dmsuite


class DeepMindEnvironment(Environment, ABC):
    """
    This abstract class is for wrapping dm_control environments
    """

    def __init__(self, dm_env, name="dmenv"):
        super().__init__()

        self.dm_env = dm_env
        self.name = name

        # Limits
        action_spec = self.dm_env.action_spec()
        obs_shape = (np.sum([array_spec.shape for array_spec in self.dm_env.observation_spec().values()]),)
        self.a_lim = np.vstack((action_spec.minimum, action_spec.maximum)).T
        self.s_dim = self.state.size
        self.a_dim = action_spec.shape[0]
        self.o_dim = obs_shape[0]

        self.render_im = None
        self.render_width = 480
        self.render_height = 480
        self.render_dt = 0.005
        self.render_skip_frames_num = 4
        self.render_frame_counter = 0

    @property
    def state(self):
        return np.atleast_2d(self.dm_env.physics.state())

    @state.setter
    def state(self, s):
        s = np.squeeze(s)
        self.dm_env.physics.set_state(s)
        mjlib.mj_step1(self.dm_env.physics.model.ptr, self.dm_env.physics.data.ptr)

    def obs(self, s):
        save_state = self.state
        self.state = s
        obs = self.obs_state
        self.state = save_state
        return obs

    def step(self, action):
        self.dm_env._reset_next_step = False  # suppress reset
        self.dm_env.step(action)
        return self.obs_state

    def render(self, mode='human'):
        self.render_frame_counter += 1
        if self.render_frame_counter > self.render_skip_frames_num:
            self.render_frame_counter = 0
        if self.render_frame_counter > 0:
            return

        frame = np.hstack([self.dm_env.physics.render(self.render_height, self.render_width, camera_id=0),
                           self.dm_env.physics.render(self.render_height, self.render_width, camera_id=1)])

        self.render_im = None
        if self.render_im is None:
            fig, ax = plt.subplots(1, 1)
            self.render_im = ax.imshow(frame)
        else:
            self.render_im.set_data(frame)
            plt.draw()

        plt.pause(self.render_dt)

    def close(self):
        self.dm_env.close()

    @property
    def obs_state(self):
        d = self.dm_env._task.get_observation(self.dm_env._physics)
        obs = np.concatenate([e for e in d.values()])
        return np.atleast_2d(obs)

    def set_num_steps_at_once(self, n):
        self.dm_env._n_sub_steps = n

    def reset(self):
        res = super().reset()
        self.render_frame_counter = -1
        return res


class PendulumDMEnvironment(DeepMindEnvironment):

    def __init__(self):
        dmenv = dmsuite.load(domain_name="pendulum", task_name="swingup")
        super().__init__(dmenv, "penddm")

        self.x0 = np.array([np.pi, 0])
        self.s_lim = np.array([[-np.pi, np.pi], [-np.inf, np.inf]])
        self.o_lim = np.array([[-1, 1], [-1, 1], [-np.inf, np.inf]])
        self.o_labels = ["x", "y", "theta dot"]

        self.task_goal = self.obs(np.zeros(2))
        self.task_cost_weights = np.array([1e2, 1e-1, 1e-1, 1e-3])
        self.task_num_rff_feats = 90
        self.task_t = 200

        self.set_num_steps_at_once(2)


class CartpoleDMEnvironment(DeepMindEnvironment):

    def __init__(self):
        dmenv = dmsuite.load(domain_name="cartpole", task_name="swingup")
        super().__init__(dmenv, "cartpoledm")

        self.x0 = np.array([0, np.pi, 0, 0])
        self.s_lim = np.array([[-1.6, 1.6], [0, 2 * np.pi], [-4, 4], [-4, 4]])
        self.o_lim = np.array([[-1.804, 1.804], [-1, 1], [-1, 1], [-10, 10], [-10, 10]])
        self.o_bounds = np.array([[-1.6, 1.6], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        self.o_labels = ["x", "xp", "yp", "x dot", "theta dot"]

        self.task_goal = self.obs(np.zeros(4))
        self.task_cost_weights = np.array([1e2, 1e2, 1e-1, 1e-1, 1e-1, 1e-1])
        self.task_num_rff_feats = 80
        self.task_t = 100

        self.set_num_steps_at_once(2)
