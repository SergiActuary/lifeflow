import numpy as np

from lifeflow._inforce_nb import _compute_inf, _compute_exits

METHODS = {"udd": 0, "cf": 1}


class Inforce:
    """In-force and per-cause exit grids for a portfolio.

    Takes any number of competing decrements. Both grids are computed on
    first access and cached; asking for a different exit method recomputes.
    """

    def __init__(self, portfolio, decrements):
        self.portfolio     = portfolio
        self.decrements    = decrements
        self._qs           = None
        self._inf          = None
        self._exits        = None
        self._exits_method = None

    def _build_qs(self):
        if self._qs is None:
            self._qs = np.stack([
                d.grid(self.portfolio) if hasattr(d, "grid") else np.asarray(d)
                for d in self.decrements
            ], axis=0)
        return self._qs

    def _mask(self):
        # todas las hipótesis comparten el mismo timeline, así que la máscara
        # de cualquiera vale para la cartera entera
        return self.decrements[0].timeline.mask(self.portfolio)

    @property
    def inf(self):
        """P(in force) at the end of each period. Shape N × T.

        Combines survival across all decrements with contract vigency, so it
        drops to zero once a policy has expired rather than levelling off.
        """
        if self._inf is None:
            self._inf = _compute_inf(self._build_qs()) * self._mask()
        return self._inf

    def exit_by(self, decrement=None, method="udd"):
        """Exits during each period attributable to one cause. Shape N × T.

        With no argument returns every cause stacked as K × N × T. These are
        unconditional probabilities measured from the start of the projection,
        so they multiply cash-flow grids directly.

        `method` is how the period's exits are split between competing causes:
        "udd" (uniform distribution of decrements) or "cf" (constant force).
        """
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {tuple(METHODS)}, got {method!r}"
            )

        if self._exits_method != method:
            n = -(-len(self.decrements) // 2)
            x, w = np.polynomial.legendre.leggauss(n)
            s = 0.5 * (x + 1.0)
            w = 0.5 * w

            self._exits        = _compute_exits(self._build_qs(), self.inf, METHODS[method], s, w)
            self._exits_method = method

        if decrement is None:
            return self._exits
        return self._exits[self.decrements.index(decrement)]
