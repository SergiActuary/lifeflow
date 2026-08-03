import inspect

import numpy as np


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


def grid(portfolio, timeline, *, jit=False, payable="post"):
    """Extend a per-policy cash flow to the whole book. Returns N × T.

    The decorated function is written for one policy at one instant: `t` runs
    from 1 to T, and every argument after it is matched by name against a
    portfolio column. Returning a tuple yields one grid per element.

    Runs as plain Python by default and accepts any flow, tuple returns
    included. `jit=True` compiles the flow with numba — far faster on large
    books after a one-off compilation, but only for purely arithmetic
    functions returning a single grid.

    Every flow is trimmed to each policy's vigency automatically: a `post`
    flow (the default) runs through the policy's term; `payable="pre"` marks a
    flow collected at the start of the period, whose last payment therefore
    falls one period earlier. Writing the amount itself — `t + 1` where the
    product calls for it — stays with the caller.
    """
    if payable not in PAYABLE:
        raise ValueError(f"payable must be one of {PAYABLE}, got {payable!r}")

    T = timeline.horizon(portfolio)
    N = len(next(iter(portfolio.cols.values())))

    pre = payable == "pre"
    mask = timeline.mask(portfolio)
    if pre:
        vigencia = np.hstack([mask[:, 1:], np.zeros((N, 1), dtype=bool)])
    else:
        vigencia = mask

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
                return out * vigencia

        else:
            def compute_row(n):
                scalars = [portfolio.cols[p][n] for p in params]
                return [f(t, *scalars) for t in range(1, T + 1)]

            def wrapper():
                results = [compute_row(n) for n in range(N)]
                arr = np.array(results)
                if arr.ndim == 3:
                    grids = tuple(arr[:, :, k] for k in range(arr.shape[2]))
                    return tuple(g * vigencia for g in grids)
                return arr * vigencia

        return wrapper

    return decorator
