import numba as nb
import numpy as np


@nb.njit(cache=True)
def _cf_share(q_vec, j):
    mu_total = 0.0
    for m in range(len(q_vec)):
        mu_total += -np.log(1.0 - q_vec[m])

    if mu_total == 0.0:
        return 0.0

    return -np.log(1.0 - q_vec[j]) / mu_total


@nb.njit(cache=True)
def _udd_share(q_vec, j, s, w):
    total = 0.0
    for k in range(len(s)):
        product = q_vec[j]
        for m in range(len(q_vec)):
            if m != j:
                product *= 1.0 - s[k] * q_vec[m]
        total += w[k] * product

    return total


@nb.njit(parallel=True, cache=True)
def _compute_inf(qs):
    k = qs.shape[0]
    n = qs.shape[1]
    t = qs.shape[2]
    result = np.zeros((n, t))

    for idx_n in nb.prange(n):
        for idx_t in range(t):
            prev = result[idx_n, idx_t - 1] if idx_t > 0 else 1.0
            stay = 1.0
            for idx_k in range(k):
                stay *= 1.0 - qs[idx_k, idx_n, idx_t]
            result[idx_n, idx_t] = prev * stay

    return result


@nb.njit(parallel=True, cache=True)
def _compute_exits(qs, inf, method_id, s, w):
    k = qs.shape[0]
    n = qs.shape[1]
    t = qs.shape[2]
    results = np.zeros((k, n, t))

    for idx_n in nb.prange(n):
        for idx_t in range(t):
            prev = inf[idx_n, idx_t - 1] if idx_t > 0 else 1.0
            q_vec = qs[:, idx_n, idx_t]

            if method_id == 0:
                for j in range(k):
                    results[j, idx_n, idx_t] = prev * _udd_share(q_vec, j, s, w)

            else:
                stay = 1.0
                for m in range(k):
                    stay *= 1.0 - q_vec[m]
                exits = 1.0 - stay
                for j in range(k):
                    results[j, idx_n, idx_t] = prev * exits * _cf_share(q_vec, j)

    return results


# warmup
_qs = np.zeros((2, 2, 2))
_inf = np.zeros((2, 2))
_s = np.array([0.5])
_w = np.array([1.0])
_compute_inf(_qs)
_compute_exits(_qs, _inf, 0, _s, _w)
del _qs, _inf, _s, _w
