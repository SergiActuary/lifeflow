import inspect

import numpy as np
from joblib import Parallel, delayed


def extend_t(T):
    def decorator(func):
        def wrapper(*args):
            result = np.zeros(T)
            for t in range(1, T + 1):
                result[t - 1] = func(t, *args)
            return result
        return wrapper
    return decorator


def grid(portfolio, timeline, *, n_jobs=-1, jit=False):
    T = timeline.horizon(portfolio)
    N = len(next(iter(portfolio.cols.values())))

    def decorator(f):
        params = list(inspect.signature(f).parameters.keys())[1:]  # skip 't'

        if jit:
            import numba as nb
            f_vec = nb.vectorize(f)
            t_grid = np.ascontiguousarray(
                np.tile(np.arange(1, T + 1, dtype=np.int64), (N, 1))
            )
            p_grids = [
                np.ascontiguousarray(
                    np.repeat(portfolio.cols[p].astype(np.float64)[:, None], T, axis=1)
                )
                for p in params
            ]

            def wrapper():
                return f_vec(t_grid, *p_grids)

        else:
            def compute_row(n):
                scalars = [portfolio.cols[p][n] for p in params]
                return [f(t, *scalars) for t in range(1, T + 1)]

            def wrapper():
                results = Parallel(n_jobs=n_jobs)(
                    delayed(compute_row)(n) for n in range(N)
                )
                arr = np.array(results)
                if arr.ndim == 3:
                    return tuple(arr[:, :, k] for k in range(arr.shape[2]))
                return arr

        return wrapper

    return decorator
