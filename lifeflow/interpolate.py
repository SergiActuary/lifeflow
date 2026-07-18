import numpy as np

METHODS = ("linear", "geomdecr", "geomincr")
DIRECTIONS = ("down", "up")


def interpolate(values, method="linear", freq=12, to="down"):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if to not in DIRECTIONS:
        raise ValueError(f"to must be one of {DIRECTIONS}, got {to!r}")

    values = np.asarray(values, dtype=float)

    if to == "down":
        if method == "linear":
            result = np.repeat(values / freq, freq)
        elif method == "geomdecr":
            result = np.repeat(1 - (1 - values) ** (1 / freq), freq)
        elif method == "geomincr":
            result = np.repeat((1 + values) ** (1 / freq) - 1, freq)
    else:
        if values.size % freq:
            raise ValueError(
                f"to='up' needs len(values) divisible by freq={freq}, got {values.size}"
            )
        if method == "linear":
            result = values.reshape(-1, freq).sum(axis=1)
        elif method == "geomdecr":
            result = 1 - (1 - values.reshape(-1, freq)).prod(axis=1)
        elif method == "geomincr":
            result = (1 + values.reshape(-1, freq)).prod(axis=1) - 1

    return result
