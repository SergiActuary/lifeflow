import numpy as np


class Hypothesis:
    def __init__(self, values, timeline, index_var=None):
        self.values = values
        self.timeline = timeline
        self.index_var = index_var

    def resolve(self, portfolio):
        if self.index_var is None:
            return 0
        return portfolio.cols[self.index_var].astype(int)

    def span(self, portfolio):
        offset = self.resolve(portfolio)
        return len(self.values) - np.max(offset)

    def grid(self, portfolio):
        offset = self.resolve(portfolio)
        ts = self.timeline.grid(portfolio)
        if np.isscalar(offset):
            g = self.values[ts]
        else:
            g = self.values[offset[:, None] + ts]
        return g * self.timeline.mask(portfolio)
