"""
Test integrands for  I(f) = E[f(Z)],  Z ~ N(0, I_s),  together with the two
wrappers needed to make the Chopin-Gerber (2022) estimators attack the same
integral on [0,1]^s.

Two routes to a CG baseline
---------------------------
(A) Rosenblatt / inverse-cdf transform.   Since Z_i = Phi^{-1}(U_i),

        E[f(Z)] = int_{[0,1]^s} f(Phi^{-1}(u)) du =: I(g),   g = f o Phi^{-1}.

    Feed g to `cubic_strat.core.estimate` (the non-vanishing estimator (9)).
    This is the cleanest head-to-head: identical integral, and CG's own
    evaluation count n = 3k^s.
    g is generally NOT in C^r([0,1]^s) with bounded derivatives, because
    Phi^{-1} blows up at the boundary.
(B) CG's own recipe for R^s integrals (their Prop. 1, Section 3.3): write
    I = int_{R^s} G(x) dx with G(x) = f(x) phi(x), and push it to [0,1]^s
    through the Student-like map psi_s, giving a vanishing integrand for
    `cubic_strat.core.vanish_estimates`.

Every integrand takes an (N, s) array and returns an (N,) array.
"""

import numpy as np
from scipy.stats import norm
from scipy.special import erfc

SQ2PI = np.sqrt(2.0 * np.pi)


# integrands

def _col(u):
    return u if u.ndim > 1 else u[:, None]


class Target:
    def __init__(self, name, f, truth, s, smooth, label):
        self.name, self.f, self.truth, self.s = name, f, truth, s
        self.smooth = smooth     # largest r with f in G^r(gamma), informally
        self.label = label       # for plot titles

    def __call__(self, z):
        return self.f(_col(np.asarray(z, dtype=float)))


def make_targets(s):
    """Dictionary of test problems in dimension s."""
    t = {}

    t['sin'] = Target(
        'sin',
        lambda z: np.sum(np.sin(z + 1.0), axis=1),
        s * np.sin(1.0) * np.exp(-0.5), s,
        np.inf, r'$f=\sum_i \sin(z_i+1)$  (analytic)')

    t['invquad'] = Target(
        'invquad',
        lambda z: np.sum(1.0 / (1.0 + z ** 2), axis=1),
        s * np.sqrt(np.pi / 2.0) * np.exp(0.5) * erfc(1.0 / np.sqrt(2.0)), s,
        np.inf, r'$f=\sum_i (1+z_i^2)^{-1}$  ($C^\infty$, bounded)')

    t['exp'] = Target(
        'exp',
        lambda z: np.sum(np.exp(z), axis=1),
        s * np.exp(0.5), s,
        np.inf, r'$f=\sum_i e^{z_i}$  (unbounded derivatives)')

    t['relu3'] = Target(
        'relu3',
        lambda z: np.sum(np.maximum(z, 0.0) ** 3, axis=1),
        s * 2.0 / SQ2PI, s,
        3, r'$f=\sum_i (z_i)_+^3$  ($\partial^3\in L^2$, $\partial^4\notin$)')

    t['abs'] = Target(
        'abs',
        lambda z: np.sum(np.abs(z), axis=1),
        s * np.sqrt(2.0 / np.pi), s,
        1, r'$f=\sum_i |z_i|$  ($\partial^1\in L^2$, $\partial^2\notin$)')

    # Shifted versions of the two non-smooth targets.  The kink of |z| and of
    # (z)_+^3 sits at z = 0, which under BOTH transforms lands on u = 1/2 --
    # a *cell boundary* of the CG stratification whenever k is even.  A
    # singularity sitting exactly on a cell boundary is never straddled by a
    # cell, so CG sees a locally smooth function and shows an artificially good
    # rate.  These shifted targets break that alignment and are the honest test.
    from scipy.integrate import quad as _quad
    _shift = 0.3

    def _tr(g):
        val, _ = _quad(lambda z: g(z) * norm.pdf(z), -15, 15, limit=500)
        return s * val

    t['relu3_shift'] = Target(
        'relu3_shift',
        lambda z: np.sum(np.maximum(z - _shift, 0.0) ** 3, axis=1),
        _tr(lambda z: max(z - _shift, 0.0) ** 3), s,
        3, r'$f=\sum_i (z_i-0.3)_+^3$  (kink off-grid)')

    t['abs_shift'] = Target(
        'abs_shift',
        lambda z: np.sum(np.abs(z - _shift), axis=1),
        _tr(lambda z: abs(z - _shift)), s,
        1, r'$f=\sum_i |z_i-0.3|$  (kink off-grid)')

    if s >= 2:
        t['sin_mixed'] = Target(
            'sin_mixed',
            lambda z: np.sin(np.sum(z, axis=1) + 1.0),
            np.sin(1.0) * np.exp(-0.5 * s), s,
            np.inf, r'$f=\sin(\sum_i z_i+1)$  (analytic, non-separable)')

    return t


# (A) inverse-cdf wrapper for CG's non-vanishing estimator

def invcdf_phi(target, clip=1e-15):
    """g(u) = f(Phi^{-1}(u)) on (0,1)^s, ready for cubic_strat.core.estimate."""
    def phi(u):
        u = _col(np.asarray(u, dtype=float))
        u = np.clip(u, clip, 1.0 - clip)
        return target(norm.ppf(u))
    return phi


# (B) CG's psi_s recipe for the vanishing estimator

def psi_map(u, tau=1.5):
    """psi_1(u) = (2u-1) / (u^tau (1-u)^tau), applied coordinatewise."""
    return (2.0 * u - 1.0) / (u ** tau * (1.0 - u) ** tau)


def psi_jac(u, tau=1.5):
    """psi_1'(u) = 2/(u^tau(1-u)^tau) + tau(2u-1)^2/(u^{tau+1}(1-u)^{tau+1})."""
    a = u ** tau * (1.0 - u) ** tau
    return 2.0 / a + tau * (2.0 * u - 1.0) ** 2 / (a * u * (1.0 - u))


def vanishing_phi(target, tau=1.5, xmax=40.0):
    """f_{G,psi}(u) with G(x) = f(x) * phi_{0,I}(x); integrates to I(f).

    Values are set to 0 where |psi| exceeds xmax in any coordinate: there the
    Gaussian factor is below 1e-300 and the product 0 * inf would be a NaN.
    """
    def phi_fun(u):
        u = _col(np.asarray(u, dtype=float))
        N = u.shape[0]
        ok = np.all((u > 1e-12) & (u < 1.0 - 1e-12), axis=1)
        out = np.zeros(N)
        if not ok.any():
            return out
        uu = u[ok]
        x = psi_map(uu, tau)
        good = np.all(np.abs(x) < xmax, axis=1)
        if not good.any():
            return out
        xg = x[good]
        jac = np.prod(psi_jac(uu[good], tau), axis=1)
        dens = np.exp(-0.5 * np.sum(xg ** 2, axis=1)) / (SQ2PI ** xg.shape[1])
        idx = np.where(ok)[0][good]
        out[idx] = target(xg) * dens * jac
        return out
    return phi_fun


# a smoothness ladder: the cleanest way to test the predicted r-dependence

def relu_family(s, powers=(1, 2, 3, 4, 5), shift=0.3):
    """f_p(z) = sum_i (z_i - shift)_+^p.

    (z-c)_+^p has p-1 continuous derivatives; its p-th derivative is p! 1{z>c},
    which is bounded hence in L^2(gamma), while the (p+1)-th is a Dirac and is
    not.  So  f_p in G^p(gamma) \\ G^{p+1}(gamma):  the Hermite smoothness index
    is exactly r = p.  Varying p therefore sweeps r while changing nothing else,
    which is what you want for checking the predicted exponent -1/2 - r/(2s).

    shift != 0 keeps the singular point off the CG cell boundaries (u = 1/2).
    """
    from scipy.integrate import quad
    out = {}
    for p in powers:
        val, _ = quad(lambda z, p=p: max(z - shift, 0.0) ** p * norm.pdf(z),
                      shift, 20, limit=500)
        out['relu%d' % p] = Target(
            'relu%d' % p,
            (lambda z, p=p: np.sum(np.maximum(z - shift, 0.0) ** p, axis=1)),
            s * val, s, p,
            r'$f=\sum_i (z_i-%.1f)_+^{%d}$  ($r=%d$)' % (shift, p, p))
    return out


# stress families: the benchmark integrands were bounded, flat in the tails and
# infinitely smooth.  These break each of those in turn.

def tail_family(s):
    """Integrands that GROW in the tails -- the region the Hermite basis
    resolves worst.  All have finite variance under gamma (required for the
    estimator to have finite variance at all)."""
    from scipy.integrate import quad
    out = {}

    out['exp1'] = Target(
        'exp1', lambda z: np.sum(np.exp(z), axis=1),
        s * np.exp(0.5), s, np.inf,
        r'$f=\sum_i e^{z_i}$  (mild tail growth)')

    out['exp2'] = Target(
        'exp2', lambda z: np.sum(np.exp(2.0 * z), axis=1),
        s * np.exp(2.0), s, np.inf,
        r'$f=\sum_i e^{2z_i}$  (strong tail growth)')

    a = 0.15   # needs a < 1/4 for E[f^2] < infinity
    out['gauss_tail'] = Target(
        'gauss_tail', lambda z: np.sum(np.exp(a * z ** 2), axis=1),
        s / np.sqrt(1.0 - 2.0 * a), s, np.inf,
        r'$f=\sum_i e^{0.15 z_i^2}$  (near-Gaussian tail growth)')

    out['invquad'] = Target(
        'invquad', lambda z: np.sum(1.0 / (1.0 + z ** 2), axis=1),
        s * np.sqrt(np.pi / 2.0) * np.exp(0.5) * erfc(1.0 / np.sqrt(2.0)),
        s, np.inf, r'$f=\sum_i (1+z_i^2)^{-1}$  (bounded, heavy shoulders)')

    v, _ = quad(lambda z: np.sqrt(abs(z)) * norm.pdf(z), -15, 15, limit=400)
    out['sqrt'] = Target(
        'sqrt', lambda z: np.sum(np.sqrt(np.abs(z)), axis=1),
        s * v, s, 0.5, r'$f=\sum_i |z_i|^{1/2}$  (cusp, $r=1/2$)')

    return out


def nonseparable_family(s):
    """Integrands that do not split across coordinates, so the tensor structure
    of the quadrature and of the truncation set cannot be exploited."""
    from scipy.integrate import quad
    out = {}

    out['sin_sum'] = Target(
        'sin_sum', lambda z: np.sin(np.sum(z, axis=1) + 1.0),
        np.sin(1.0) * np.exp(-0.5 * s), s, np.inf,
        r'$f=\sin(\sum_i z_i + 1)$  (analytic, non-separable)')

    sd = np.sqrt(s)
    v, _ = quad(lambda t: 1.0 / (1.0 + t ** 2) * norm.pdf(t, scale=sd),
                -20 * sd, 20 * sd, limit=600)
    out['invquad_sum'] = Target(
        'invquad_sum', lambda z: 1.0 / (1.0 + np.sum(z, axis=1) ** 2),
        v, s, np.inf, r'$f=(1+(\sum_i z_i)^2)^{-1}$  (non-separable)')

    w, _ = quad(lambda t: max(t - 0.3, 0.0) ** 3 * norm.pdf(t, scale=sd),
                -20 * sd, 20 * sd, limit=600)
    out['relu3_sum'] = Target(
        'relu3_sum', lambda z: np.maximum(np.sum(z, axis=1) - 0.3, 0.0) ** 3,
        w, s, 3, r'$f=(\sum_i z_i-0.3)_+^3$  (non-separable, $r=3$)')

    return out

