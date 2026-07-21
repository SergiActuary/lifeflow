import numpy as np


class Timeline:
    """The projection horizon, taken from a portfolio column.

    `contract_boundary` names the column holding each policy's term. The
    longest policy sets T; the others are cut off by `mask`.
    """

    def __init__(self, contract_boundary):
        self.contract_boundary = contract_boundary

    def resolve(self, portfolio) -> np.ndarray:
        """Each policy's term, as integers."""
        return portfolio.cols[self.contract_boundary].astype(int)

    def horizon(self, portfolio) -> int:
        """T: the longest term in the book."""
        return self.resolve(portfolio).max()

    def grid(self, portfolio) -> np.ndarray:
        """The time axis, 0 to T-1, for indexing rate vectors."""
        return np.arange(self.horizon(portfolio))

    def mask(self, portfolio) -> np.ndarray:
        """N × T of booleans: True while each policy is still in force."""
        end = self.resolve(portfolio)
        T = end.max()
        return np.arange(T) < end[:, None]
