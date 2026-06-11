# lifeflow

Vectorized life actuarial engine for cash-flow projection and BEL computation.

Built on pure numpy — no loops, no Excel, no proprietary software.

---

## Why lifeflow

Traditional actuarial tools (Prophet, MoSes, Excel) project policies one by one. lifeflow projects an entire portfolio at once as a single **N × T matrix operation**, where N is the number of policies and T is the time horizon.

The result: a portfolio of 50,000 policies with 60-year monthly projections runs in under 2 seconds. A full million-policy BEL with batching completes in around 30 seconds.

---

## Key features

- **Fully vectorized** — every calculation is a numpy matrix operation over the entire portfolio
- **Exact multiple-decrement formulas** — inclusion-exclusion under UDD for any number of competing causes (mortality, lapse, disability, …)
- **Composable objects** — `Portfolio`, `Timeline`, `Hypothesis`, `Inforce`, `Flow` share a common `.grid(portfolio)` interface that returns N × T arrays
- **Rate conversion** — `annual_to_monthly` / `monthly_to_annual` under constant-force assumption
- **BEL-ready** — discount externally (EIOPA curve or any vector), multiply and sum

---

## Installation

```bash
pip install lifeflow
```

---

## Quick start

```python
import numpy as np
import lifeflow as lf

# Portfolio: polars or numpy-backed table of policies
portfolio = lf.Portfolio(df, id_col="POLICY_ID")

# Hypothesis: indexed rate vector (e.g. monthly qx by age)
qx = lf.Hypothesis(values=qx_monthly, timeline=tl, index_var="age_months")
wx = lf.Hypothesis(values=wx_monthly, timeline=tl, index_var="age_months")

# Timeline: defines T from the portfolio's contract boundary column
tl = lf.Timeline("contract_end_months")

# Inforce: tPx grid with exact competing-decrement exits
inforce = lf.Inforce([qx, wx])

inf_grid      = inforce.grid(portfolio)          # N × T  survival probabilities
exit_death    = inforce.exit_by(qx, portfolio)   # N × T  exits by mortality
exit_lapse    = inforce.exit_by(wx, portfolio)   # N × T  exits by lapse

# Flow: any formula over portfolio columns and time
death_benefit = lf.Flow(lambda DEATH_PAY_SINGLE: DEATH_PAY_SINGLE, tl)

# BEL: multiply grids and discount externally
v   = (1 / 1.03) ** (np.arange(T) / 12)         # EIOPA-style monthly discount
bel = (exit_death.grid(portfolio) * death_benefit.grid(portfolio) * v).sum(axis=1)
```

---

## Core objects

| Object | What it does |
|---|---|
| `Portfolio` | Holds policy data; exposes columns as numpy arrays |
| `Timeline` | Defines T = max(contract boundary) + 1 and the per-policy mask |
| `Hypothesis` | N × T grid of rates, looked up by age, duration or any index variable |
| `Inforce` | N × T survival grid (tPx); exact inclusion-exclusion exits per cause |
| `Flow` | N × T cash-flow grid from a Python formula over portfolio columns |

All objects share the same interface: `.grid(portfolio) → np.ndarray`.

---

## Performance

| Portfolio | Time horizon | Tool | Time |
|---|---|---|---|
| 50,000 policies | 720 months | lifeflow | ~1.5 s |
| 50,000 policies | 720 months | Prophet FIS | ~10–30 min |
| 1,000,000 policies | 720 months | lifeflow (batched) | ~30 s |

Peak RAM scales as `grids × N × T × 4 bytes` (float32). A batch of 50k policies at T=720 uses ~1 GB.

---

## Design principles

- **No proprietary dependencies** — numpy + polars only
- **Auditability first** — every intermediate grid is a plain numpy array, inspectable at any step
- **Actuarial correctness** — exact UDD multiple-decrement formula, not the 1/2 approximation
- **Discount is external** — lifeflow handles hypotheses; the EIOPA curve is your input

---

## License

MIT
