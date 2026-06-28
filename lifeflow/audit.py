import calendar
from datetime import date

import polars as pl


def _add_months(dt, n):
    month = dt.month - 1 + n
    year  = dt.year + month // 12
    month = month % 12 + 1
    day   = min(dt.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _parse_date(s):
    d, m, y = s.split("/")
    return date(int(y), int(m), int(d))


def audit_grid(grid, portfolio, id_col=None, present_date=None):
    N = grid.shape[0]

    if id_col is not None:
        ids      = portfolio.cols[id_col]
        row_name = id_col
    else:
        ids      = list(range(N))
        row_name = "poliza"

    if grid.ndim == 1:
        return pl.DataFrame({row_name: ids, "value": grid})

    T = grid.shape[1]

    if present_date is not None:
        base = _parse_date(present_date) if isinstance(present_date, str) else present_date
        cols = [_add_months(base, t + 1).strftime("%d/%m/%Y") for t in range(T)]
    else:
        cols = [f"t{t}" for t in range(T)]

    return pl.DataFrame({row_name: ids, **{c: grid[:, t] for t, c in enumerate(cols)}})
