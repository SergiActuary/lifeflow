"""Implementación de referencia del reparto UDD por inclusión-exclusión.

Este código fue el núcleo de lifeflow/_inforce_nb.py hasta que se sustituyó
por cuadratura de Gauss-Legendre. Se conserva aquí, fuera de la librería,
como referencia independiente contra la que validar el kernel nuevo.

Ambos métodos resuelven la misma integral:

    (aq)_j = q_j * int_0^1 prod_{m != j} (1 - s*q_m) ds

La implementación de abajo la resuelve analíticamente: desarrolla el producto
en sus 2^(K-1) términos e integra cada monomio, lo que produce los signos
alternados y los denominadores 1/(|S|+1). El kernel de la librería evalúa la
misma integral por cuadratura, que es exacta porque el integrando es un
polinomio de grado K-1.

Al ser dos caminos de cálculo distintos sobre la misma definición, un error
en cualquiera de los dos rompe la coincidencia.
"""

import numba as nb
import numpy as np


@nb.njit
def udd_correction(q_vec, j):
    """Factor por el que las causas competidoras reducen las salidas de j.

    Suma sobre los subconjuntos S de las causas distintas de j:

        sum_S (-1)^|S| / (|S|+1) * prod_{m in S} q_m

    El subconjunto vacío aporta el 1 de la inicialización.
    """
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
def compute_exits_udd(qs, inf):
    """Reparto UDD de las salidas, por inclusión-exclusión. Devuelve (K, N, T)."""
    k = qs.shape[0]
    n = qs.shape[1]
    t = qs.shape[2]
    results = np.zeros((k, n, t))

    for idx_n in nb.prange(n):
        for idx_t in range(t):
            prev = inf[idx_n, idx_t - 1] if idx_t > 0 else 1.0
            q_vec = qs[:, idx_n, idx_t]
            for j in range(k):
                results[j, idx_n, idx_t] = q_vec[j] * prev * udd_correction(q_vec, j)

    return results
