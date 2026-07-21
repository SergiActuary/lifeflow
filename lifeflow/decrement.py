import numpy as np


class Decrement:
    """A rate vector — mortality, lapse, disability — on the projection frequency.

    `index_var` names the portfolio column each policy enters the table at,
    typically age in months; without it every policy reads the table from its
    first element.
    """

    def __init__(self, values, timeline, index_var=None):
        self.values = values
        self.timeline = timeline
        self.index_var = index_var

    def resolve(self, portfolio) -> int | np.ndarray:
        """Each policy's entry point into the rate vector."""
        if self.index_var is None:
            return 0
        return portfolio.cols[self.index_var].astype(int)

    def grid(self, portfolio) -> np.ndarray:
        """N × T of rates, zero once each policy's contract has expired.

        Raises IndexError if the table runs out before the horizon — that
        means the rate vector is too short for the oldest entry age.
        """
        offset = self.resolve(portfolio)
        ts = self.timeline.grid(portfolio)
        if isinstance(offset, np.ndarray):
            grd = self.values[offset[:, None] + ts]
        else:
            grd = self.values[ts]
        return grd * self.timeline.mask(portfolio)
