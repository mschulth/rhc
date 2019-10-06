# Implementation of Receding Horizon Curiosity

## Installation

It is easiest to setup a virtual environment in order to install the required site-packages without modifying your global python installation. We are using Python3 (to be precise 3.6.8) and hence (assuming the code from this repository is in [DIR]), the following lines of code setup the virtualenv and install required packages:

```bash
cd [DIR]
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

This will install OpenAI Gym, casadi, dm\_control, and stable\_baselines in the required versions.

For running experiments you need MuJoCo because the environments of dm_control are used. Be sure you installed MuJoCo correctly to the default directory, otherwise follow the installation tutorial from [dm_control](https://github.com/deepmind/dm_control).

Then you can install dm_control by using the follwing command:

```bash
pip install git+git://github.com/deepmind/dm_control.git
```

In case of problems with the installation of dm\_control, the setup instructions of [dm_control](https://github.com/deepmind/dm_control) are quite helpful.

## Usage

The exploration in the pendulum environment can be launched by 

```bash
python3 run_exploration.py --env pendulum --method rhc_us --n_episodes 20
python3 run_exploration.py --env pendulum --method infogain --n_episodes 20
python3 run_exploration.py --env pendulum --method prederr --n_episodes 20
python3 run_exploration.py --env pendulum --method random --n_episodes 20
python3 visualise_results.py
```
The first four commands run different exploration methods on the pendulum experiment, namely Receding Horizon Curiosity in the uncertainty sampling formulation, SAC using the infogain and prediction error objective, and exploration using uniformly random actions. Evaluations are automatically done after each episode and written in pickle files to the results folder.
For RHC and random actions, additional plots of the planned and executed trajectories are stored after each episodes in  respective subfolders of the results folder.
The last command visualises the results of all res_\*.pkl files from the results folder.
