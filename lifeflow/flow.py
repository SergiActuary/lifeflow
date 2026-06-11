import inspect

import numpy as np


class Flow:
    def __init__(self, formula, timeline):
        self.formula = formula
        self.timeline = timeline

    def grid(self, portfolio, include_present=False) -> np.ndarray:
        ts = self.timeline.grid(portfolio)
        names = inspect.signature(self.formula).parameters
        args = [ts if n == "t" else portfolio.cols[n][:, None] for n in names]
        cashflow = self.formula(*args)
        result = cashflow * self.timeline.mask(portfolio)
        if not include_present:
            result[:, 0] = 0
        return result
