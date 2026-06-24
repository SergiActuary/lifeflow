import numpy as np


def decrements(**kwargs):
    # convierte y valida
    arrays = {name: np.asarray(v, dtype=float) for name, v in kwargs.items()}
    lengths = [len(v) for v in arrays.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"All decrements must have the same length, got {lengths}")

    return {"array": np.stack(list(arrays.values())), "names": list(arrays.keys())}
