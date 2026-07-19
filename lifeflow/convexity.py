import numpy as np

def convexity(vec, interest):
    t = np.arange(1, len(interest) + 1)
    discount = (1 + interest) ** -t
    pv = (vec * discount).sum()
    derivative_2 = (t * (t + 1) * vec * discount / ((1 + interest) ** 2)).sum()
    result = derivative_2 / pv

    return result
