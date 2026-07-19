import numpy as np

def duration_macaulay(vec, interest):
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    w_t = (vec * discount) / pv
    result = (t * w_t).sum()

    return result
