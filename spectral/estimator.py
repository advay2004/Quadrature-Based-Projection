"""
Hermite-projection / Gauss-Hermite control-variate estimators for

        I(f) = E[f(Z)],   Z ~ N(0, I_s).

Estimators implemented
----------------------
 plain      IhatM(f)   = (1/K) sum_i [ f(Z_i) - sum_{0<|a|<M} bhat_a hhat_a(Z_i) ]
 antithetic IhatM^A(f) = (1/K) sum_i [ (f(Z_i)+f(-Z_i))/2
                                       - sum_{0<|a|<M, |a| even} bhat_a hhat_a(Z_i) ]

with bhat_a from a tensorised m-point Gauss-Hermite rule.

Evaluation budget
-----------------
 plain:       n = m^s + K
 antithetic:  n = m^s + 2K
"""


import numpy as np
from numpy.polynomial.hermite_e import hermegauss


# orthonormal Hermite polynomials

def he_orthonormal(z, M):
    """hhat_n(z) = He_n(z)/sqrt(n!) for n = 0,...,M-1.

    Stable three-term recurrence
        hhat_{n+1}(z) = ( z hhat_n(z) - sqrt(n) hhat_{n-1}(z) ) / sqrt(n+1).

    Parameters
    ----------
    z : (N,) array
    M : int, number of polynomials (degrees 0..M-1)

    Returns
    -------
    (N, M) array
    """
    z = np.asarray(z, dtype=float)
    out = np.empty((z.shape[0], M))
    out[:, 0] = 1.0
    if M > 1:
        out[:, 1] = z
    for n in range(1, M - 1):
        out[:, n + 1] = (z * out[:, n] - np.sqrt(n) * out[:, n - 1]) / np.sqrt(n + 1)
    return out


def he_orthonormal_scaled(z, M):
    """hhat_n(z) * exp(-z^2/4).

    Same recurrence, rescaled initial conditions.  This factor is what keeps
    the Gauss-Hermite nodes usable: at the extreme nodes (|x| ~ 2 sqrt(m))
    hhat_n(x) overflows, while hhat_n(x) exp(-x^2/4) stays O(1).
    """
    z = np.asarray(z, dtype=float)
    sc = np.exp(-0.25 * z ** 2)
    out = np.empty((z.shape[0], M))
    out[:, 0] = sc
    if M > 1:
        out[:, 1] = z * sc
    for n in range(1, M - 1):
        out[:, n + 1] = (z * out[:, n] - np.sqrt(n) * out[:, n - 1]) / np.sqrt(n + 1)
    return out


def gauss_hermite_1d(m, M=None):
    """m-point Gauss-Hermite rule for the standard normal measure, returned in
    a form that stays accurate for large m.

    Nodes: eigenvalues of the Jacobi matrix of the probabilists' recurrence
    (zero diagonal, off-diagonal sqrt(n)). These are accurate.

    Weights: Not the squared first eigenvector components -- those lose all
    relative accuracy for the tiny outer weights since they bottom out around 1e-60
    when the true value is e.g. 1e-350).  We use the
    Christoffel formula instead,

        w_q = 1 / sum_{n<m} hhat_n(x_q)^2 = exp(-x_q^2/2) / S_q,
        S_q = sum_{n<m} [hhat_n(x_q) exp(-x_q^2/4)]^2,

    where every quantity on the right is O(1) or smaller.

    Returns
    -------
    x : (m,) nodes
    w : (m,) true weights, sum 1, exact on degree 2m-1
    v : (m,) compensated weights  v_q = w_q exp(x_q^2/4)
    G : (m, M) scaled basis  G[q, n] = hhat_n(x_q) exp(-x_q^2/4)   (M = m if None)
    """
    from scipy.linalg import eigh_tridiagonal
    x = eigh_tridiagonal(np.zeros(m), np.sqrt(np.arange(1, m)),
                         eigvals_only=True)
    Gfull = he_orthonormal_scaled(x, m)
    S = np.sum(Gfull ** 2, axis=1)
    ok = S > 0.0            # outer nodes underflow entirely, their true weight
    v = np.zeros(m)         # ~exp(-x^2/2) < 1e-300, so dropping them is exact
    w = np.zeros(m)         # double precision
    v[ok] = np.exp(-0.25 * x[ok] ** 2) / S[ok]
    w[ok] = np.exp(-0.50 * x[ok] ** 2) / S[ok]
    M = m if M is None else M
    return x, w, v, Gfull[:, :M]


# tensor-grid helpers

def tensor_grid(x1d, s):
    """All m^s tensor nodes as an (m^s, s) array, C-order over (i1,...,is)."""
    mesh = np.meshgrid(*([x1d] * s), indexing='ij')
    return np.stack([a.ravel() for a in mesh], axis=1)


def total_degree_mask(M, s):
    """Boolean tensor of shape (M,)*s, True where 0 < |alpha| < M."""
    idx = np.indices((M,) * s).reshape(s, -1).sum(axis=0).reshape((M,) * s)
    mask = (idx > 0) & (idx < M)
    return mask, idx


# quadrature coefficients

def quad_coefficients(f, s, m, M, return_nevals=False):
    """bhat_alpha for all alpha in {0,...,M-1}^s, by tensorised m-point GH.

        bhat_alpha = sum_q w_q f(xi_q) hhat_alpha(xi_q)

    Requires M <= m for the coefficients to be free of the worst aliasing
    where each alpha_i <= m-1, so f*hhat_alpha is integrated by a rule exact on
    degree 2m-1 whenever f is itself a polynomial of degree <= m-1.

    Returns
    -------
    b : (M,)*s array
    nevals : m^s  (only if return_nevals)
    """
    if M > m:
        raise ValueError("need M <= m (aliasing control)")
    x1d, _, v1d, H1 = gauss_hermite_1d(m, M)
    pts = tensor_grid(x1d, s)
    fv = np.asarray(f(pts), dtype=float).reshape((m,) * s)

    Wv = v1d
    for _ in range(s - 1):
        Wv = np.multiply.outer(Wv, v1d)
    A = fv * Wv
    # contract every axis in turn; tensordot appends the new axis at the end,
    # so after s contractions the axes are back in the original order.
    for _ in range(s):
        A = np.tensordot(A, H1, axes=([0], [0]))
    if return_nevals:
        return A, m ** s
    return A


def eval_expansion(b, Z, M):
    """sum_alpha b_alpha hhat_alpha(Z_i) for each sample i.

    b : (M,)*s array (already masked: unwanted alphas set to 0)
    Z : (K, s) array
    """
    K, s = Z.shape
    Hs = [he_orthonormal(Z[:, i], M) for i in range(s)]
    R = b.reshape(M, -1)                       # (M, M^{s-1})
    A = Hs[0] @ R                              # (K, M^{s-1})
    for i in range(1, s):
        A = A.reshape(K, M, -1)
        A = np.einsum('kaj,ka->kj', A, Hs[i])
    return A.ravel()


# the estimators

def hermite_estimate(f, s, m, M, K, antithetic=True, rng=None,
                     coeffs=None, return_parts=False):
    """One realisation of the (antithetic) spectral control-variate estimator.

    Parameters
    ----------
    f : callable, (N, s) -> (N,)
    s : dimension
    m : Gauss-Hermite nodes per axis (quadrature cost m^s)
    M : truncation degree, keep 0 < |alpha| < M (requires M <= m)
    K : number of Monte Carlo draws
    antithetic : use (f(Z)+f(-Z))/2 and even-degree corrections only
    coeffs : precomputed (M,)*s coefficient tensor (reused across replications)

    Returns
    -------
    dict with 'est', 'nevals'
    """
    rng = np.random.default_rng() if rng is None else rng
    if coeffs is None:
        coeffs = quad_coefficients(f, s, m, M)
    b = coeffs.copy()

    mask, deg = total_degree_mask(M, s)
    keep = mask.copy()
    if antithetic:
        keep &= (deg % 2 == 0)
    b[~keep] = 0.0                              

    Z = rng.standard_normal((K, s))
    corr = eval_expansion(b, Z, M)
    if antithetic:
        val = 0.5 * (f(Z) + f(-Z)) - corr
        nevals = m ** s + 2 * K
    else:
        val = f(Z) - corr
        nevals = m ** s + K
    est = float(np.mean(val))
    out = {'est': est, 'nevals': nevals, 'm': m, 'M': M, 'K': K}
    if return_parts:
        out['per_sample_var'] = float(np.var(val, ddof=1))
    return out


def budget_split(n, s, theta=0.5, antithetic=True, M_rule=None, m_max=None):
    """Turn a total evaluation budget n into (m, M, K).

    theta : fraction of the budget spent on the deterministic quadrature grid
            (theta * n  evaluations  ->  m = floor((theta n)^{1/s}))
    M_rule: callable m -> M.  Default M = m (the largest value allowed by the
            aliasing condition M <= m).
    """
    m = int(np.floor((theta * n) ** (1.0 / s)))
    m = max(m, 2)
    if m_max is not None:
        m = min(m, m_max)
    M = m if M_rule is None else max(1, min(m, int(M_rule(m))))
    rest = n - m ** s
    per_draw = 2 if antithetic else 1
    K = max(1, int(rest // per_draw))
    return m, M, K


# exact variance decomposition


def reference_coefficients(f, s, D, m_ref=None):
    """High-accuracy b_alpha for |alpha_i| < D, used as a stand-in for the exact
    Hermite coefficients.  Uses a much finer GH rule than the estimator does."""
    if m_ref is None:
        m_ref = max(4 * D, 200)
    return quad_coefficients(f, s, m_ref, D)


def variance_decomposition(b_ref, bhat, M, s, antithetic=True):
    """K * Var of the estimator, split into truncation and aliasing parts.

        K * Var =  sum_{|a| >= M} b_a^2      (truncation / tail of the series)
                 + sum_{0<|a|<M} (b_a - bhat_a)^2   (aliasing of retained coeffs)

    restricted to even |a| in the antithetic case.  b_ref must be at least as
    large as bhat in every axis.
    """
    D = b_ref.shape[0]
    _, deg_ref = total_degree_mask(D, s)
    sel = np.ones_like(deg_ref, dtype=bool)
    sel[(0,) * s] = False
    if antithetic:
        sel &= (deg_ref % 2 == 0)

    trunc = float(np.sum(b_ref[sel & (deg_ref >= M)] ** 2))

    pad = np.zeros_like(b_ref)
    sl = tuple(slice(0, M) for _ in range(s))
    pad[sl] = bhat
    inner = sel & (deg_ref > 0) & (deg_ref < M)
    alias = float(np.sum((b_ref[inner] - pad[inner]) ** 2))
    return {'truncation': trunc, 'aliasing': alias, 'total': trunc + alias}
