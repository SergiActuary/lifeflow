import inspect

import numpy as np
from joblib import Parallel, delayed


def extend_t(func):
    def grid_method(T, *args):
        result = np.zeros(T)
        for t in range(T):
            result[t] = func(t, *args)
        return result
    func.grid = grid_method
    return func


def grid(func=None, *, n_jobs=-1):
    def decorator(f):
        params = list(inspect.signature(f).parameters.keys())[1:]  # skip 't'

        def grid_method(portfolio, timeline):
            T = timeline.horizon(portfolio)
            N = len(next(iter(portfolio.cols.values())))

            def compute_row(n):
                scalars = [portfolio.cols[p][n] for p in params]
                return [f(t, *scalars) for t in range(T)]

            results = Parallel(n_jobs=n_jobs)(
                delayed(compute_row)(n) for n in range(N)
            )
            return np.array(results)

        f.grid = grid_method
        return f

    if func is not None:
        return decorator(func)
    return decorator
