import polars as pl


class Portfolio:
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
        self._require_id()
        return self.cols[self.id_col]

    def selectid(self, value):
        self._require_id()
        return Portfolio(self.df.filter(self.df[self.id_col] == value), self.id_col)
