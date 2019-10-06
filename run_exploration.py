from env.EnvironmentFactory import EnvironmentFactory
from exp.ExplorationFactory import ExplorationFactory
from model.RFFModel import RFFModel
from util.Evaluation import DatasetEvaluation, TaskEvaluation, EvaluationCollection
import logging
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env", required=True, help="name of the environment", choices=EnvironmentFactory.get_options())
ap.add_argument("-m", "--method", required=True, help="name of the exploration method",
                choices=ExplorationFactory.get_options())
ap.add_argument("-s", "--seed", type=int, required=False, help="seed for the execution", default=2)
ap.add_argument("-n", "--n_episodes", type=int, required=False, help="number of episodes", default=20)
args = vars(ap.parse_args())

"""args = {
    'env': "pendulum",
    'method': 'rhc_us',
    'seed': 2,
    'n_episodes': 20
}"""

env_str = args['env']
exp_method = args['method']
seed = args['seed']
n_episodes = args['n_episodes']
save_results = True
save_plots = True

# define logging behavior
logger = logging.getLogger("exploration")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# create environment
env = EnvironmentFactory.create(env_str)
model = RFFModel(env.x_dim, env.o_dim, env.task_num_rff_feats)

# evaluation settings
eval_ds = DatasetEvaluation(model, env)
eval_pl = TaskEvaluation(model, env)
eval_pl.seed = seed
eval = EvaluationCollection(eval_ds, eval_pl)

# create exploration runner
exp = ExplorationFactory.create(exp_method, env, model, evaluation=eval)
exp.seed = seed
exp.n_episodes = n_episodes
exp.save_plots = save_plots

# run exploration
exp.run()

# save result
if save_results:
    exp.save_results()
