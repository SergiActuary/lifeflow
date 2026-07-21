import polars as pl


class Portfolio:
    """A book of policies, one row per policy.

    Every column is exposed as a numpy array in `cols`, which is where `@grid`
    and `Decrement` look up policy variables by name. `id_col` is optional and
    only needed for `ids` and `selectid`.
    """

    def __init__(self, df, id_col=None):
        self.df = df
        self.id_col = id_col
        self.cols = {name: df[name].to_numpy() for name in df.columns}

    def _require_id(self):
        if self.id_col is None:
            raise ValueError(
                "This Portfolio has no id_col. Create it with "
                "Portfolio(df, id_col='NPOL') to use ids or selectid."
            )

    @property
    def ids(self):
        """The identifier column as an array."""
        self._require_id()
        return self.cols[self.id_col]

    def selectid(self, value):
        """A new Portfolio holding only the policy with that id.

        Useful to project a single policy in isolation when debugging. An id
        that matches nothing yields an empty Portfolio rather than an error.
        """
        self._require_id()
        return Portfolio(self.df.filter(self.df[self.id_col] == value), self.id_col)
