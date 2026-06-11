import numpy as np


def annual_to_monthly(annual) -> np.ndarray:
    """Annual decrement rates -> monthly (constant force). Length x -> x*12."""
    monthly = 1 - (1 - np.asarray(annual, dtype=float)) ** (1 / 12)
    return np.repeat(monthly, 12)


def monthly_to_annual(monthly) -> np.ndarray:
    """Monthly decrement rates -> annual (constant force). Length x*12 -> x."""
    monthly = np.asarray(monthly, dtype=float)
    if len(monthly) % 12 != 0:
        raise ValueError(
            f"monthly vector length must be a multiple of 12, got {len(monthly)}"
        )
    return 1 - np.prod(1 - monthly.reshape(-1, 12), axis=1)
