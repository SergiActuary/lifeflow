import numba as nb
import numpy as np


@nb.njit
def _udd_correction(q_vec, j):
    n_others = len(q_vec) - 1
    others = np.empty(n_others)
    idx = 0
    for m in range(len(q_vec)):
        if m != j:
            others[idx] = q_vec[m]
            idx += 1

    correction = 1.0

    for mask in range(1, 1 << n_others):
        size = 0
        product = 1.0

        for b in range(n_others):
            if mask & (1 << b):
                size += 1
                product *= others[b]

        sign = 1 if (size % 2 == 0) else -1
        correction += sign / (size + 1) * product

    return correction


@nb.njit(parallel=True)
def _compute_all(qs, method_id):
    k = qs.shape[0]
    n = qs.shape[1]
    t = qs.shape[2]
    results = np.zeros((k + 1, n, t))

    for idx_n in nb.prange(n):
        results[0, idx_n, 0] = 1.0
        for idx_t in range(1, t):
            stay = 1.0
            for idx_k in range(k):
                stay *= 1.0 - qs[idx_k, idx_n, idx_t]
            results[0, idx_n, idx_t] = results[0, idx_n, idx_t - 1] * stay
            q_vec = qs[:, idx_n, idx_t]
            for j in range(k):
                results[1 + j, idx_n, idx_t] = (
                    q_vec[j] * results[0, idx_n, idx_t - 1] * _udd_correction(q_vec, j)
                )
    return results


# warmup
_qs = np.zeros((2, 2, 2))
_compute_all(_qs, 0)
del _qs
