import numpy as np

def duration_macaulay(vec, interest):
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    w_t = (vec * discount) / pv
    result = (t * w_t).sum()

    return result

def duration_hicks(vec, interest):
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    derivative = (t * vec * (discount/(1 + interest))).sum()
    result = derivative / pv

    return result

def convexity(vec, interest):
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    derivative_2 = (t * (t + 1) * vec * discount / ((1 + interest) ** 2)).sum()
    result = derivative_2 / pv

    return result
