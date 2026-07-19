# lifeflow

Python library for life actuarial cash-flow calculations and BEL.

Built on numpy and numba — all calculations are vectorized as N × T grids, where N is the number of policies and T is the time horizon. The decrement engine runs in parallel across policies via JIT compilation.

---

## What it does

lifeflow lets the actuary define their own cash flows as plain Python functions. The library then automates:

- **tINFx derivation** — policy in-force grid from any combination of decrements, with exact multiple-decrement formulas under UDD
- **Exit grids** — per-cause exit rates, computed once and cached alongside the in-force
- **Decrements** — mortality, lapse, disability or any rate vector, indexed by age, duration or any policy variable
- **Portfolio extension** — cash flows written for a single policy broadcast automatically across the whole portfolio

The result of every object is a plain numpy array of shape N × T, inspectable at any step.

---

## Installation

```bash
pip install lifeflow
```

---

## Usage

```python
import numpy as np
import polars as pl
import lifeflow as lf

df = pl.DataFrame({
    "POLICY_ID":     [1, 2, 3],
    "age_months":    [480, 540, 600],
    "term_months":   [120, 180, 240],
    "death_capital": [100_000.0, 250_000.0, 50_000.0],
})

portfolio = lf.Portfolio(df, id_col="POLICY_ID")
tl        = lf.Timeline("term_months")
```

`Timeline` takes the column holding each policy's contract boundary. It sets the horizon `T` (the longest policy) and masks every grid past each policy's own end.

Rates already on the projection frequency are wrapped as decrements, indexed by a policy variable:

```python
qx = lf.Decrement(qx_monthly, tl, index_var="age_months")
wx = lf.Decrement(wx_monthly, tl, index_var="age_months")
```

Splitting annual rates to a monthly grid is a one-line numpy operation and is left to the caller — `1 − (1−q)**(1/12)` repeated over each year for a decrement, `(1+i)**(1/12) − 1` for a rate that compounds upward.

`Inforce` derives the in-force grid and the exits by cause. Both are computed lazily and cached:

```python
inforce = lf.Inforce(portfolio, [qx, wx])

tINFx      = inforce.inf          # N × T
exit_death = inforce.exit_by(qx)  # N × T
exit_lapse = inforce.exit_by(wx)  # N × T
```

`exit_by()` with no argument returns all causes stacked as K × N × T.

Cash flows are written for a single policy at a single instant. The `@grid` decorator extends them to the full N × T portfolio — every parameter after `t` is matched by name against a portfolio column:

```python
@lf.grid(portfolio, tl, jit=True)
def death_benefit(t, death_capital):
    return death_capital

@lf.grid(portfolio, tl, jit=True)
def discount(t):
    return 1.0 / (1.0 + 0.03) ** (t / 12)

bel = (exit_death * death_benefit() * discount()).sum(axis=1)
```

**`t` runs from 1 to T, not from 0.** The present is never a projection period: the first column is one period into the future, already discounted. Every part of the library follows this — `@grid`, `@extend_t`, and the duration functions in `alm`.

Use `jit=True` for purely mathematical flows: the function is compiled with `numba.vectorize` and runs over the grid with no Python loop. For flows that need arbitrary Python (lookups, branching on objects), leave the default `jit=False` and the rows are computed in parallel via joblib (`n_jobs=-1`). A flow returning a tuple yields one N × T grid per element.

To build a single 1-D vector of length T rather than a portfolio grid, use `@extend_t`:

```python
@lf.extend_t(240)
def v(t, rate):
    return 1.0 / (1.0 + rate) ** t

v(0.03)  # shape (240,)
```

Any grid can be dumped to a polars DataFrame for inspection, with real dates as column headers:

```python
lf.audit_grid(tINFx, portfolio, id_col="POLICY_ID", present_date="31/12/2025")
```

---

## Notes

Exits are split under a **uniform distribution of decrements (UDD)** assumption, using the exact inclusion-exclusion correction rather than an approximation. The number of terms grows as `2^(K-1)` per cause, so `Inforce` warns above 6 decrements.

---

## License

MIT
