import numpy as np


class Hypothesis:
    def __init__(self, values, timeline, index_var=None):
        self.values = values
        self.timeline = timeline
        self.index_var = index_var

    def resolve(self, portfolio) -> int | np.ndarray:
        if self.index_var is None:
            return 0
        return portfolio.cols[self.index_var].astype(int)

    def span(self, portfolio) -> int:
        offset = self.resolve(portfolio)
        return len(self.values) - np.max(offset)

    def grid(self, portfolio) -> np.ndarray:
        offset = self.resolve(portfolio)
        ts = self.timeline.grid(portfolio)
        ts_lookup = np.maximum(ts - 1, 0)
        if isinstance(offset, np.ndarray):
            grid = self.values[offset[:, None] + ts_lookup]
        else:
            grid = self.values[ts_lookup]
        grid = grid * self.timeline.mask(portfolio)
        grid[:, 0] = 0
        return grid
