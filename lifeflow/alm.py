import numpy as np

def duration_macaulay(vec, interest):
    """Macaulay duration of a cash flow, in periods.

    The present-value-weighted average of the payment times. `interest` are
    spot rates by maturity, on the same frequency and length as `vec`.
    """
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    w_t = (vec * discount) / pv
    result = (t * w_t).sum()

    return result

def duration_hicks(vec, interest):
    """Modified (Hicks) duration, in periods.

    Sensitivity of the present value to a parallel shift of the whole curve,
    so no single yield is needed: each term carries its own discount rate.
    Flattens to the textbook D/(1+i) when the curve is flat.
    """
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    derivative = (t * vec * (discount/(1 + interest))).sum()
    result = derivative / pv

    return result

def convexity(vec, interest):
    """Convexity of a cash flow, in periods squared.

    Second derivative of the present value under the same parallel shift as
    `duration_hicks`. Note the units: dividing by 12 gives months, but
    converting this to years takes 144.
    """
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    derivative_2 = (t * (t + 1) * vec * discount / ((1 + interest) ** 2)).sum()
    result = derivative_2 / pv

    return result
