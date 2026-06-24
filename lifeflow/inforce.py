import numpy as np


class Inforce:
    def __init__(self, decrements, method="cf"):
        self.array = decrements["array"]
        self.names = decrements["names"]
        self.method = method

    @property
    def inforce(self):
        return np.cumprod(np.prod(1 - self.array, axis=0))

    def exit_by(self, name):
        idx = self.names.index(name)
        d_j = self.array[idx]
        prev = np.insert(self.inforce, 0, 1.0)[:-1]

        if self.method == "cf":
            mu_j = -np.log(1 - d_j)
            mu_total = -np.log(1 - self.array).sum(axis=0)
            return prev * (mu_j / mu_total) * (1 - np.exp(-mu_total))

        elif self.method == "udd":
            pass  # lo implementamos después
