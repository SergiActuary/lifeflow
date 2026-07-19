import numpy as np

def duration_hicks(vec, interest):
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    derivative = (t * vec * (discount/(1 + interest))).sum()
    result = derivative / pv

    return result
