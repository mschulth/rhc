from exp.Exploration import Exploration, MaxEpisodesReachedException
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGFeedForwardPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
import gym
from gym import spaces
from abc import ABC, abstractmethod
import numpy as np
import logging


class MFRLExploration(Exploration):
    """
    Model-free reinforcement learning exploration runner
    """

    def __init__(self, env, model, name=None, evaluation=None, objective=None):
        if objective is None:
            raise AttributeError("An objective must be provided. Can be be 'infogain' and 'prederr'.")
        else:
            self.objective = objective

        if name is None:
            name = self.objective
        super().__init__(env, model, name, evaluation)

        self.seed = 0
        self.rl_algo = "sac"

    def run(self):
        self._init()

        env = self.env
        model = self.model
        objective = self.objective

        if objective == "infogain":
            wenv = InfogainEnv(env, model)
        elif objective == "prederr":
            wenv = PrederrEnv(env, model)
        else:
            raise AttributeError("Objective '{}' is unknown. Needs to be 'infogain' or 'prederr'".format(objective))

        wenv.max_episode_len = self.horizon
        wenv.end_episode_callback = self._end_episode
        dvenv = DummyVecEnv([lambda: wenv])

        if self.rl_algo == "ddpg":
            self.logger.info("Setting up DDPG as model-free RL algorithm.")
            pn = AdaptiveParamNoiseSpec()
            an = NormalActionNoise(np.array([0]), np.array([1]))
            rl_model = DDPG(DDPGMlpPolicy, dvenv, verbose=1, render=False, action_noise=an, param_noise=pn,
                            nb_rollout_steps=self.horizon, nb_train_steps=self.horizon)
        elif self.rl_algo == "sac":
            self.logger.info("Setting up SAC as model-free RL algorithm.")
            rl_model = SAC(SACMlpPolicy, dvenv, verbose=1, learning_starts=self.horizon)
        else:
            raise AttributeError("Model-free RL algorithm '{}' is unknown.".format(self.rl_algo))

        # Train the agent
        max_steps_total = self.horizon * self.n_episodes * 100
        try:
            self.logger.info("Start the agent")
            rl_model.learn(total_timesteps=max_steps_total, seed=self.seed)
        except MaxEpisodesReachedException:
            print("Exploration finished.")


class DDPGCustomPolicy(DDPGFeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(DDPGCustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[32, 32],
                                               layer_norm=False,
                                               feature_extraction="mlp")


class MFRLEnv(gym.Env, ABC):

    def __init__(self, env, model, logger=None):
        super().__init__()
        self.benv = env
        self.model = model

        self.action_space = spaces.Box(low=env.a_lim[:, 0], high=env.a_lim[:, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=env.o_lim[:, 0], high=env.o_lim[:, 1], dtype=np.float32)

        self.step_counter = 0
        self.episode_counter = 1
        self.max_episode_len = None
        self.end_episode_callback = None

        if logger is None:
            self.logger = logging.getLogger("exploration")
        else:
            self.logger = logger

    @abstractmethod
    def reward_step(self, action, step):
        """
        This method executes a step in the environment and adds the transition to the model
        :param action: the action to execute
        :param step: the step number
        :return: next state observation, reward, next state valid
        """
        return None, 0, True

    def step(self, action):
        if np.any(np.isnan(action)):
            raise AttributeError()

        self.step_counter += 1
        self.logger.info("Step {}".format(self.step_counter))

        action = np.atleast_1d(action)
        obs, reward, valid = self.reward_step(action, self.step_counter)

        done = (not valid) or self.step_counter >= self.max_episode_len
        if done:
            self.logger.info("End episode in step {}".format(self.step_counter))
            # create empty dummy features to model call end episode callback
            dummy_feat_x = np.empty((0, self.benv.x_dim))
            dummy_feat_y = np.empty((0, self.benv.o_dim))
            if self.end_episode_callback is not None:
                self.end_episode_callback(dummy_feat_x, dummy_feat_y)
            self.episode_counter += 1

        return obs, reward, done, {}

    def reset(self):
        print("Reset environment in step {}".format(self.step_counter))
        self.step_counter = 0
        return self.benv.reset()

    def render(self, mode='human', close=False):
        return self.benv.render()

    def create_features(self, a, obs_before, obs_after):
        a = np.atleast_2d(a)
        x = np.hstack((obs_before, a))
        y = obs_after - obs_before
        return x, y


class InfogainEnv(MFRLEnv):

    def reward_step(self, action, step):
        ent_before = self.model.param_entropy()
        obs_before = self.benv.obs_state
        obs = self.benv.step(action)
        obs_after = self.benv.obs_state
        valid = np.all(self.benv.state_valid(obs_after))

        if valid:
            x, y = self.create_features(action, obs_before, obs_after)
            # add data without updating the model so than the model remains unchanged for the current episode
            self.model.add_data(x, y, refine_model=True)

        ent_after = self.model.param_entropy()
        reward = ent_before - ent_after  # reward is reduction of entropy

        return obs, reward, valid


class PrederrEnv(MFRLEnv):

    def reward_step(self, action, step):
        # execute step
        obs_before = self.benv.obs_state
        obs = self.benv.step(action)
        obs_after = self.benv.obs_state
        x, y = self.create_features(action, obs_before, obs_after)

        # compute reward
        pred = self.model.predict(x, ret_var=False)
        reward = np.sqrt(np.mean((pred - y)**2))

        valid = np.all(self.benv.state_valid(obs_after))  # not valid = end of episode
        if valid:
            self.model.add_data(x, y, refine_model=False)

        return obs, reward, valid
