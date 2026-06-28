import numpy as np


class Timeline:
    def __init__(self, contract_boundary):
        self.contract_boundary = contract_boundary

    def resolve(self, portfolio) -> np.ndarray:
        return portfolio.cols[self.contract_boundary].astype(int)

    def horizon(self, portfolio) -> int:
        return self.resolve(portfolio).max()

    def grid(self, portfolio) -> np.ndarray:
        return np.arange(self.horizon(portfolio))

    def mask(self, portfolio) -> np.ndarray:
        end = self.resolve(portfolio)
        T = end.max()
        return np.arange(T) < end[:, None]
