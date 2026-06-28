import warnings

import numpy as np

from lifeflow._inforce_nb import _compute_inf, _compute_exits


class Inforce:
    def __init__(self, portfolio, decrements):
        self.portfolio  = portfolio
        self.decrements = decrements
        self._qs        = None
        self._inf       = None
        self._exits     = None

    def _build_qs(self):
        if self._qs is None:
            self._qs = np.stack([
                d.grid(self.portfolio) if hasattr(d, "grid") else np.asarray(d)
                for d in self.decrements
            ], axis=0)
        return self._qs

    @property
    def inf(self):
        if self._inf is None:
            self._inf = _compute_inf(self._build_qs())
        return self._inf

    def exit_by(self, decrement=None, method="udd"):
        if self._exits is None:
            if len(self.decrements) > 6:
                warnings.warn(
                    f"Inforce: UDD with {len(self.decrements)} decrements requires "
                    f"{2 ** (len(self.decrements) - 1)} inclusion-exclusion terms per cause — "
                    f"consider a constant-force method instead",
                    stacklevel=2,
                )
            method_id = {"udd": 0}.get(method, 0)
            self._exits = _compute_exits(self._build_qs(), self.inf, method_id)
        if decrement is None:
            return self._exits
        return self._exits[self.decrements.index(decrement)]
