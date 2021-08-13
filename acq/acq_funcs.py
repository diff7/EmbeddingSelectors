import numpy as np
from acq.k_center import k_center_greedy


class CoreFuncs:
    def __init__(self, name, all_data_size, data_path=None):

        self.all_data_size = all_data_size
        self.name = name
        self.initial_random_size = 10
        self.data_path = data_path

    def sample(self, sample_size, features):
        if self.name == "random":
            return self.random_sample(sample_size)
        if self.name == "k_center":
            return self.k_center_sample(sample_size, features)
        if self.name == "forgettable":
            return self.forgettable(sample_size, features)

    def random_sample(self, sample_size):
        return np.random.choice(self.all_data_size, sample_size, replace=False)

    def k_center_sample(self, sample_size, features):
        # half = sample_size // 2
        initial_random = self.random_sample(self.initial_random_size)

        k_centered = k_center_greedy(
            features, initial_random, sample_size - self.initial_random_size
        )
        return list(initial_random) + list(k_centered)

    def forgettable(self, sample_size, features):
        path = ["/"] + self.data_path.split("/")[:-1] + ["forgetable.npy"]

        scores = np.load("/".join(path))

        assert len(features) == len(
            scores
        ), "Num scores does not match with dataset size"

        indices = np.argsort(scores)[::-1]
        return indices[:sample_size]