import numpy as np
import pickle
import os
import logging
import os.path


class DatasetHandler:

    def __init__(self, env, logger=None):
        self.env = env
        self.n = 10000  # number of datapoints
        self.seed = 0  # seed for creation of the data set
        self.path = "data"
        self.prefix = "data"

        if logger is None:
            self.logger = logging.getLogger("exploration")
        else:
            self.logger = logger

    def trans_file(self):
        file = os.path.join(self.path, self.prefix + "_{}_{}.pkl".format(self.env.name, self.n))
        return file

    def traj_file(self):
        file = os.path.join(self.path, self.prefix + "_t_{}_{}_{}.pkl".format(self.env.name, self.t, self.n))
        return file

    def create_random_trajectories(self, t):
        """
        Creates self.n random trajectories
        :param t: the length (num of time steps) of the trajectories to create
        :return: feature inputs and targets
        """
        env = self.env
        s_dim = env.s_dim
        o_dim = env.o_dim
        a_dim = env.a_dim
        x_dim = env.x_dim
        s_lim = env.s_lim
        a_lim = env.a_lim
        sa_lim = np.vstack((s_lim, a_lim))
        assert(x_dim == o_dim + a_dim)

        # create randomly sampled states
        rand = np.random.rand(self.n, t, s_dim + a_dim)
        limdist = sa_lim[:, 1] - sa_lim[:, 0]
        tmp = rand * limdist[None, None, :]
        x = tmp + sa_lim[:, 0][None, None, :]
        x[:, 1:, :s_dim] = np.nan

        # arrays to store transition data
        x_o = np.empty((self.n, t, x_dim)) * np.nan  # observations + actions
        y_o = np.empty((self.n, t, o_dim)) * np.nan  # difference to next observation

        # execute trajectories to get data
        for i in range(self.n):
            for j in range(t):
                si = x[i, j, :s_dim]
                ai = x[i, j, -a_dim:]

                # set state, execute step and observe next state
                env.state = si
                sio = env.obs_state
                env.step(ai)
                sn = env.state
                sno = env.obs_state

                if j + 1 < t:
                    x[i, j + 1, :s_dim] = sn

                x_o[i, j, :o_dim] = sio
                x_o[i, j, -a_dim:] = ai
                y_o[i, j, :] = sno - sio

        return x_o, y_o

    def create_traj_dataset(self):
        """
        Creates a trajectory data set and saves it to self.path using the prefix self.prefix.
        :param basename: a tag that will be part of the filename
        :param t: the length of the trajectory
        """
        x, y = self.create_random_trajectories(self.t)
        d = {'x': x, 'y': y}
        file = self.traj_file()
        self.logger.info("Trajectory dataset created. Save data to '{}'.".format(file))
        with open(file, 'wb') as f:
            pickle.dump(d, f)

    def create_trans_dataset(self):
        """
        Creates a trajectory data set and saves it to self.path using the prefix self.prefix.
        :param basename: a tag that will be part of the filename
        """
        x, y = self.create_random_trajectories(1)
        x = np.reshape(x, (np.shape(x)[0], np.shape(x)[2]))
        y = np.reshape(y, (np.shape(y)[0], np.shape(y)[2]))
        d = {'x': x, 'y': y}
        file = self.trans_file()
        self.logger.info("Transition dataset created. Save data to '{}'.".format(file))
        with open(file, 'wb') as f:
            pickle.dump(d, f)

    @staticmethod
    def get_data_set_train(env):
        dc = DatasetHandler(env)
        dc.n = 10000
        file = dc.trans_file()
        if not os.path.isfile(file):
            dc.seed = 1
            dc.create_trans_dataset()

        with open(file, 'rb') as f:
            d = pickle.load(f)
        x = d['x'][:2000, :]
        y = d['y'][:2000, :]
        test_x = d['x'][2000:, :]
        test_y = d['y'][2000:, :]
        test_xy = np.concatenate((test_x, test_y), axis=-1)

        return x, y, test_xy

    @staticmethod
    def get_data_set_test(env):
        dc = DatasetHandler(env)
        dc.n = 10000
        dc.t = 10
        file = dc.traj_file()
        if not os.path.isfile(file):
            dc.seed = 1
            dc.create_traj_dataset()

        with open(file, 'rb') as f: d = pickle.load(f)
        test_x = d['x']
        test_y = d['y']
        test_xy = np.concatenate((test_x, test_y), axis=-1)
        return test_xy
