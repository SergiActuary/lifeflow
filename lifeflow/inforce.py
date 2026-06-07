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
        others = sum(d.grid(portfolio) for d in self.decrements if d is not decrement)
        return inforce * q * (1 - 0.5 * others)
