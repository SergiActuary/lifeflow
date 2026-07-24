# lifeflow

Python library for life actuarial cash-flow calculations and BEL.

Built on numpy and numba — all calculations are vectorized as N × T grids, where N is the number of policies and T is the time horizon. The decrement engine is JIT-compiled with numba, and cash flows can be compiled too.

A worked end-to-end example — computing the BEL of a portfolio of mixed endowments — is published at **<https://sergiactuary.github.io/lifeflow/>** (source in [`docs/example.qmd`](docs/example.qmd)).

---

## What it does

lifeflow lets the actuary define their own cash flows as plain Python functions. The library then automates:

- **tINFx derivation** — policy in-force grid from any combination of decrements
- **Exit grids** — per-cause exit rates under UDD or constant force, exact for any number of causes, computed once and cached alongside the in-force
- **Decrements** — mortality, lapse, disability or any rate vector, indexed by age, duration or any policy variable
- **Portfolio extension** — cash flows written for a single policy broadcast automatically across the whole portfolio
- **Interest-rate sensitivity** — Macaulay and Hicks duration, and convexity, over a spot curve

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

`Probabilities` derives the in-force grid and the exits by cause. Both are computed lazily and cached:

```python
prob = lf.Probabilities(portfolio, [qx, wx])

tINFx      = prob.inforce       # N × T
exit_death = prob.exit_by(qx)   # N × T
exit_lapse = prob.exit_by(wx)   # N × T
```

`inforce` is P(in-force): survival across all decrements (stochastic) combined with contract vigency (deterministic), so it is zero once the policy has expired.

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

Use `jit=True` for purely mathematical flows: the function is compiled with `numba.vectorize` and runs over the grid with no Python loop. For flows that need arbitrary Python (lookups, branching on objects), leave the default `jit=False` and the rows are computed as plain sequential Python. A flow returning a tuple yields one N × T grid per element (supported on the default `jit=False` path).

### Prepayable flows

A prepayable flow is collected at the *start* of the period, so its last payment falls one period before the contract expires. The formula stays yours — write `t + 1` if that is what the product does — but the shortened vigency does not: it depends on each policy's own `duration`, so with staggered terms the cut-off lands on a different column for every row. That is what `payable="pre"` handles:

```python
@lf.grid(portfolio, tl, payable="pre")
def premium(t):
    return 400.0 + 30.0 * t        # the t+1 shift is written by you
```

`payable="post"` is the default and leaves the grid untouched — no masking is needed there, since `inf` already carries the vigency into the product. Under `"pre"`, skipping the cut-off leaves a payment in the last vigent cell where the in-force is still positive: the result comes out plausible and roughly 40% high, with no warning.

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

### Interest-rate sensitivity

Duration and convexity are portfolio-level ALM measures, so they take the cash flow of the whole book — collapse the grid first — together with a vector of spot rates on the same frequency and of the same length:

```python
flow = (exit_death * death_benefit()).sum(axis=0)   # N × T  →  T

lf.duration_macaulay(flow, spot_monthly)   # months
lf.duration_hicks(flow, spot_monthly)      # months
lf.convexity(flow, spot_monthly)           # months²
```

Discounting is spot by maturity, `v_t = (1 + s_t)^-t`, not chained — an EIOPA curve split to monthly gives spot rates, and compounding them as if they were forwards overstates the present value.

`duration_hicks` and `convexity` differentiate with respect to a parallel shift of the whole curve, so they need no single IRR: each term is divided by its own `(1 + s_t)` once or twice. With a flat curve they collapse to the textbook `D/(1+i)` and its second derivative.

These are deliberately *not* computed per policy. Portfolio duration is not the average of policy durations — it is the present-value-weighted average — so returning a vector would invite an incorrect `.mean()`. Filter the rows you want and collapse those instead.

---

## Notes

`exit_by` splits the period's exits between the competing causes under one of two assumptions, chosen with `method`:

| method | assumption | survival within the period |
|---|---|---|
| `"udd"` (default) | uniform distribution of decrements | `1 − s·q` (linear) |
| `"cf"` | constant force | `(1−q)^s` (exponential) |

Both are exact, not approximations. Which one to use is an actuarial choice, not a numerical one: within a period you only observe who entered and who left, never when. The two agree to about `1e-6` at typical monthly rates and diverge as rates grow.

Under UDD the exit probability is

```
(aq)_j = q_j · ∫₀¹ Π_{m≠j} (1 − s·q_m) ds
```

which the engine evaluates by Gauss–Legendre quadrature. The integrand is a polynomial of degree `K−1`, so `⌈K/2⌉` nodes integrate it exactly — no error is introduced. The cost is `O(K³)` rather than the `O(K·2^K)` of the equivalent closed form, so there is no practical limit on the number of decrements.

The closed form is kept in `tests/reference_udd.py` and used to validate the engine.

---

## License

MIT
