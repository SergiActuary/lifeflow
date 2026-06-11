from itertools import combinations

import numpy as np


class Inforce:
    def __init__(self, decrements):
        self.decrements = decrements

    def grid(self, portfolio) -> np.ndarray:
        p_stay = 1.0
        for d in self.decrements:
            p_stay = p_stay * (1 - d.grid(portfolio))
        cum = np.cumprod(p_stay, axis=1)
        inforce = np.ones_like(cum)
        inforce[:, 1:] = cum[:, :-1]
        return inforce

    def exit_by(self, decrement, portfolio) -> np.ndarray:
        inforce = self.grid(portfolio)
        q = decrement.grid(portfolio)
        others = [d.grid(portfolio) for d in self.decrements if d is not decrement]

        correction = np.ones_like(q)
        for k in range(1, len(others) + 1):
            sign = (-1) ** k
            for idx in combinations(range(len(others)), k):
                product = np.ones_like(q)
                for i in idx:
                    product = product * others[i]
                correction = correction + sign / (k + 1) * product

        return inforce * q * correction
