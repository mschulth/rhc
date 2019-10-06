from exp.MFRLExploration import MFRLExploration
from exp.RandomExploration import RandomExploration
from exp.PlanningbasedExploration import MaxTrajectoryEntropyExploration, MinModelEntropyExploration


class ExplorationFactory:

    @classmethod
    def create(cls, exp_method, exp, model, evaluation=None):
        if exp_method == "rhc_us":
            exp = MaxTrajectoryEntropyExploration(exp, model, evaluation=evaluation)
        elif exp_method == "rhc_mvr":
            exp = MinModelEntropyExploration(exp, model, evaluation=evaluation)
        elif exp_method == "infogain":
            exp = MFRLExploration(exp, model, evaluation=evaluation, objective="infogain")
        elif exp_method == "prederr":
            exp = MFRLExploration(exp, model, evaluation=evaluation, objective="infogain")
        elif exp_method == "random":
            exp = RandomExploration(exp, model, name="random", evaluation=evaluation)
        else:
            raise AttributeError("Exploration Method '{}' not known.".format(exp_method))

        return exp

    @classmethod
    def get_options(cls):
        return ["rhc_us", "rhc_mvr", "infogain", "prederr", "random"]
