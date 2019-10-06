from env.MountainCarEnvironment import MountainCarEnvironment
from env.DeepMindEnvironment import PendulumDMEnvironment, CartpoleDMEnvironment


class EnvironmentFactory:

    @classmethod
    def create(cls, env_name):
        if env_name == "pendulum":
            env = PendulumDMEnvironment()
        elif env_name == "cartpole":
            env = CartpoleDMEnvironment()
        elif env_name == "mountaincar":
            env = MountainCarEnvironment()
        else:
            raise AttributeError("Environment '{}' not known.".format(env_name))

        env.reset()
        return env

    @classmethod
    def get_options(cls):
        return ["mountaincar", "pendulum", "cartpole"]
