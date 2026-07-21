import inspect

import numpy as np
from joblib import Parallel, delayed


def extend_t(T):
    """Evaluate a function over t = 1..T, returning a 1-D vector of length T.

    For curves that do not vary by policy — a discount factor, an index.
    Arguments after `t` are passed straight through on every call.
    """

    def decorator(func):
        def wrapper(*args):
            result = np.zeros(T)
            for t in range(1, T + 1):
                result[t - 1] = func(t, *args)
            return result
        return wrapper
    return decorator


PAYABLE = ("post", "pre")


def grid(portfolio, timeline, *, n_jobs=-1, jit=False, payable="post"):
    """Extend a per-policy cash flow to the whole book. Returns N × T.

    The decorated function is written for one policy at one instant: `t` runs
    from 1 to T, and every argument after it is matched by name against a
    portfolio column. Returning a tuple yields one grid per element.

    `jit=True` compiles the function with numba for purely arithmetic flows;
    the default runs them in parallel through joblib and accepts any Python.

    `payable="pre"` marks a flow collected at the start of the period, whose
    last payment therefore falls one period before the contract expires; it
    trims the vigency accordingly. Writing the amount itself — `t + 1` where
    the product calls for it — stays with the caller.
    """
    if payable not in PAYABLE:
        raise ValueError(f"payable must be one of {PAYABLE}, got {payable!r}")

    T = timeline.horizon(portfolio)
    N = len(next(iter(portfolio.cols.values())))

    # La fórmula del flujo la escribe el actuario: si es prepagable, usará t+1.
    # Lo que no puede ver a ojo es que el último pago de cada póliza cae un
    # período antes de su vencimiento, así que la vigencia se corre una columna.
    pre = payable == "pre"
    if pre:
        mask = timeline.mask(portfolio)
        vigencia = np.hstack([mask[:, 1:], np.zeros((N, 1), dtype=bool)])

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
                out = f_vec(t_grid, *p_grids)
                return out * vigencia if pre else out

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
                    grids = tuple(arr[:, :, k] for k in range(arr.shape[2]))
                    return tuple(g * vigencia for g in grids) if pre else grids
                return arr * vigencia if pre else arr

        return wrapper

    return decorator
