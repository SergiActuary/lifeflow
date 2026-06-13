import warnings

import numpy as np

from lifeflow._inforce_nb import _compute_all


class Inforce:
    def __init__(self, decrements):
        self.decrements = decrements

    def grid(self, portfolio) -> np.ndarray:
        pass

    def exit_by(self, decrement, portfolio) -> np.ndarray:
        pass

    def compute_all(self, portfolio, method="udd"):
        if len(self.decrements) > 6:
            warnings.warn(
                f"Inforce: UDD with {len(self.decrements)} decrements requires "
                f"{2 ** (len(self.decrements) - 1)} inclusion-exclusion terms per cause — "
                f"consider a constant-force method instead",
                stacklevel=2,
            )
        qs = np.stack([d.grid(portfolio) for d in self.decrements], axis=0)
        method_id = {"udd": 0}.get(method, 0)
        return _compute_all(qs, method_id)
