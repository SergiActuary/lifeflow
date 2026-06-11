# lifeflow

Object-oriented Python library for life actuarial cash-flow calculations.

Built on numpy — all calculations are vectorized as N × T grids, where N is the number of policies and T is the time horizon.

---

## What it does

lifeflow lets the actuary define their own cash flows using Python lambdas. The library then automates:

- **tINFx derivation** — policy in-force grid from any combination of decrement hypotheses, with exact multiple-decrement formulas under UDD
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

inforce    = lf.Inforce([qx, wx])
exit_death = inforce.exit_by(qx, portfolio)   # N × T

benefit = lf.Flow(lambda DEATH_PAY_SINGLE: DEATH_PAY_SINGLE, tl)

bel = (exit_death * benefit.grid(portfolio) * discount).sum(axis=1)
```

---

## License

MIT
