# lifeflow

Object-oriented Python library for life actuarial cash-flow calculations.

Built on numpy and numba — all calculations are vectorized as N × T grids, where N is the number of policies and T is the time horizon. The core decrement engine runs in parallel across policies via JIT compilation.

---

## What it does

lifeflow lets the actuary define their own cash flows using Python lambdas. The library then automates:

- **tINFx derivation** — policy in-force grid from any combination of decrement hypotheses, with exact multiple-decrement formulas under UDD
- **Exit grids** — per-cause exit rates computed in a single fused pass alongside the in-force, with no intermediate arrays
- **Actuarial hypotheses** — mortality, lapse, disability or any rate vector, indexed by age, duration or any policy variable
- **Portfolio extension** — all calculations broadcast automatically across the full portfolio

The result of every object is a plain numpy array of shape N × T, inspectable at any step.

---

## Installation

```bash
pip install lifeflow
```

---

## Usage

```python
import lifeflow as lf

portfolio = lf.Portfolio(df, id_col="POLICY_ID")
tl        = lf.Timeline("contract_end_months")

qx = lf.Hypothesis(values=qx_monthly, timeline=tl, index_var="age_months")
wx = lf.Hypothesis(values=wx_monthly, timeline=tl, index_var="age_months")

inforce = lf.Inforce([qx, wx])
results = inforce.compute_all(portfolio)  # (K+1) × N × T
tINFx      = results[0]   # in-force
exit_death = results[1]   # exits by qx
exit_lapse = results[2]   # exits by wx

benefit = lf.Flow(lambda DEATH_PAY_SINGLE: DEATH_PAY_SINGLE, tl)

bel = (exit_death * benefit.grid(portfolio) * discount).sum(axis=1)
```

---

## License

MIT
