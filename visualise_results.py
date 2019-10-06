from util.ResultHandler import ResultHandler
import matplotlib.pyplot as plt
import logging

# define options
result_dir = "results"
prefix = "res_"
postfix = ".pkl"

# define logging behavior
logger = logging.getLogger("exploration")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# load results
rh = ResultHandler()
rh.load_results(result_dir, prefix, postfix)

# likelihood plot
res_llh = rh.get_results("n_step_llh")
plt.figure()
for i in range(res_llh.shape[0]):
    plt.plot(res_llh[i], label=rh.labels[i])
plt.legend()
plt.xlabel("episode")
plt.ylabel("log likelihood")
plt.title("Random Trajectory Evaluation")
plt.show()

# task cost plot
res_llh = rh.get_results("task_cost")
plt.figure()
for i in range(res_llh.shape[0]):
    plt.plot(res_llh[i], label=rh.labels[i])
plt.legend()
plt.xlabel("episode")
plt.ylabel("cost")
plt.title("Task Evaluation")
plt.show()
