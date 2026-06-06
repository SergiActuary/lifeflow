class Portfolio:
    def __init__(self, df):
        self.df = df
        self.cols = {nombre: df[nombre].to_numpy() for nombre in df.columns}
