import numpy as np


def interpolate(values, method="udd", freq=12, to="down"):
    values = np.asarray(values, dtype=float)

    if to == "down":
        if method == "udd":
            result = np.repeat(values / freq, freq)
        elif method == "cf":
            result = np.repeat(1 - (1 - values) ** (1 / freq), freq)
        elif method == "compound":
            result = np.repeat((1 + values) ** (1 / freq) - 1, freq)
        else:
            result = "Method not recognized"
    elif to == "up":
        if method == "udd":
            result = values.reshape(-1, freq).sum(axis=1)
        elif method == "cf":
            result = 1 - (1 - values.reshape(-1, freq)).prod(axis=1)
        elif method == "compound":
            result = (1 + values.reshape(-1, freq)).prod(axis=1) - 1
        else:
            result = "Method not recognized"
    else:
        result = "Direction not recognized"

    return result
