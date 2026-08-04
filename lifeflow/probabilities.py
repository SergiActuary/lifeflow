import numpy as np

from lifeflow._probabilities_rs import compute_inf, compute_exit

METHODS = {"udd": 0, "cf": 1}


class Probabilities:
    """In-force and per-cause exit grids for a portfolio.

    Takes any number of competing decrements. `inforce` and each requested
    cause of `exit_by` are computed on first access and cached. Asking for a
    single cause never materializes the others.
    """

    def __init__(self, portfolio, decrements):
        self.portfolio  = portfolio
        self.decrements = decrements
        self._qs        = None
        self._inforce   = None
        self._exits     = {}      # caché por (índice de causa, método) -> N×T

    def _build_qs(self):
        if self._qs is None:
            self._qs = np.stack([
                d.grid(self.portfolio) if hasattr(d, "grid") else np.asarray(d)
                for d in self.decrements
            ], axis=0)
        return self._qs

    def _mask(self):
        return self.decrements[0].timeline.mask(self.portfolio)

    def _quadrature(self):
        n = -(-len(self.decrements) // 2)
        x, w = np.polynomial.legendre.leggauss(n)
        return 0.5 * (x + 1.0), 0.5 * w

    @property
    def inforce(self):
        """P(in force) at the end of each period. Shape N × T."""
        if self._inforce is None:
            self._inforce = compute_inf(self._build_qs()) * self._mask()
        return self._inforce

    def _exit_one(self, j, method):
        key = (j, method)
        if key not in self._exits:
            s, w = self._quadrature()
            self._exits[key] = compute_exit(
                self._build_qs(), self.inforce, METHODS[method], s, w, j
            )
        return self._exits[key]

    def exit_by(self, decrement=None, method="udd"):
        """Exits during each period attributable to one cause. Shape N × T.

        With no argument returns every cause stacked as K × N × T (unpackable).
        Asking for a single cause computes and caches only that cause.
        """
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {tuple(METHODS)}, got {method!r}"
            )

        if decrement is None:
            return np.stack(
                [self._exit_one(j, method) for j in range(len(self.decrements))],
                axis=0,
            )
        return self._exit_one(self.decrements.index(decrement), method)
