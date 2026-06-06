import numpy as np


class Timeline:
    def __init__(self, contract_boundary, included=True):
        self.contract_boundary = contract_boundary
        self.included = included

    def resolve(self, portfolio):
        end = portfolio.cols[self.contract_boundary].astype(int)
        if self.included:
            end = end + 1
        return end

    def horizon(self, portfolio):
        return self.resolve(portfolio).max()

    def grid(self, portfolio):
        return np.arange(self.horizon(portfolio))

    def mask(self, portfolio):
        end = self.resolve(portfolio)
        T = end.max()
        return np.arange(T) < end[:, None]
