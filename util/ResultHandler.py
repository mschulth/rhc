import os
import glob
import pickle
import numpy as np
from pathlib import Path
import logging


class ResultHandler:

    def __init__(self):
        self.logger = logging.getLogger("evaluation")
        self.results = []
        self.labels = []
        self.n = np.inf

    def get_results(self, key):
        results = np.array(list(map(lambda x: x[key][:self.n], self.results)))
        return results

    def get_results_keys(self):
        return self.results[0].keys()

    def load_results(self, path, prefix, postfix):
        labels = []
        results = []

        full_path = os.path.join(path, prefix + "*" + postfix)
        files = glob.glob(full_path)
        for file in files:
            self.logger.info("Read file {}".format(file))
            with open(file, 'rb') as f:
                c = pickle.load(f)

            label = Path(file).name[len(prefix):-len(postfix)]
            result = self.merge_dicts(c)
            self.n = min(self.n, len(c))

            labels.append(label)
            results.append(result)

        self.labels = labels
        self.results = results

    def merge_dicts(self, dicts):
        """
        Converts a list of dicts to a dict of lists.
        All dicts have to contain the same keys, otherwise missing values are omitted in the resulting lists.
        :param dicts: A list of dict
        :return: A dict of lists
        """
        res_dict = {}
        for dct in dicts:
            for key, value in dct.items():
                # res_dict[key] = res_dict.get(key, []).append(value)
                if key not in res_dict:
                    res_dict[key] = []
                res_dict[key].append(value)
        return res_dict

