"""The Pima marginal likelihood as (a) a Gaussian integral, (b) their cube integral."""

import numpy as np


def lp_vec(X, chunk=20000):
    """Vectorised log-posterior; identical to looping logpost, ~100x faster."""
    X = np.atleast_2d(X)
    d = X.shape[1]
    P = PREDS[:, :d]
    out = np.empty(X.shape[0])
    for a in range(0, X.shape[0], chunk):
        b = X[a:a + chunk]
        lin = b @ P.T
        out[a:a + chunk] = (-np.sum(np.logaddexp(0.0, -lin), axis=1)
                            - (0.5 / SCALE_PRIOR ** 2) * np.sum(b ** 2, axis=1))
    return out

# ---------------------------------------------------------------- our version
def gaussian_integrand(d, scale_prop=1.5):
    """F(z) with Z ~ N(0, I_d);  E[F(Z)] = marginal likelihood * exp(-maxlp).

    beta = mu + A z with A = scale_prop * Cu^T, so the sampling distribution of
    beta is exactly N(mu, A A^T).  F is the importance weight p(beta)/q(beta),
    which is smooth, positive and decays like exp(-(scale^2-1)||z||^2/2).
    """
    L = laplace(d)
    A = scale_prop * L['Cu'].T
    logdetA = np.sum(np.log(np.abs(np.diag(A))))
    const = 0.5 * d * np.log(2.0 * np.pi) + logdetA - L['maxlp']

    def F(z):
        z = np.atleast_2d(z)
        beta = L['mu'] + z @ A.T
        return np.exp(lp_vec(beta) + const + 0.5 * np.sum(z ** 2, axis=1))
    return F, L

# ------------------------------------------------------------- their version
def psi(u, t):
    umu = u * (1.0 - u)
    tm1 = 2.0 * u - 1.0
    umutau = umu ** t
    z = tm1 / umutau
    jac = 2.0 / umutau + t * tm1 ** 2 / umu ** (t + 1)
    return z, np.sum(np.log(jac), axis=1)

def cube_integrand(d, tau=1.0, scale_prop=1.5, variant='as_written'):
    """variant='as_written'  -> exactly pima_common.phi
       variant='prop1'       -> the sign/constant implied by their Proposition 1
    """
    L = laplace(d)
    Cu = L['Cu'] / scale_prop

    def phi(u):
        u = np.atleast_2d(u)
        z, ljac = psi(u, tau)
        x = L['mu'] + z @ Cu
        if variant == 'as_written':
            cst = 0.5 * np.sum(np.log(np.diag(Cu)))
            lw = lp_vec(x) - (ljac - cst) - L['maxlp']
        else:
            logdet = np.sum(np.log(np.abs(np.diag(Cu))))
            lw = lp_vec(x) + ljac + logdet - L['maxlp']
        return np.exp(lw)
    return phi
