"""Laplace approximation of the Pima logistic posterior, as in prep_pima.py."""
import numpy as np
from scipy import linalg, optimize
from particles import datasets as dts

PREDS = dts.Pima().data          # already y_i * x_i, shape (768, 9)
SCALE_PRIOR = 5.0

def logpost(beta, preds=None):
    p = PREDS if preds is None else preds
    d = np.shape(beta)[-1]
    lin = np.dot(p[:, :d], beta)
    loglik = np.sum(-np.log1p(np.exp(-lin)))
    logprior = -(0.5 / SCALE_PRIOR ** 2) * np.sum(beta ** 2)
    return logprior + loglik

def laplace(d):
    P = PREDS[:, :d]
    res = optimize.minimize(lambda b: -logpost(b), np.zeros(d), method='BFGS',
                            options={'maxiter': 1000, 'gtol': 1e-12})
    mu = res.x
    sig = 1.0 / (1.0 + np.exp(-(P @ mu)))
    H = (P * (sig * (1 - sig))[:, None]).T @ P + np.eye(d) / SCALE_PRIOR ** 2
    Sigma = linalg.inv(H)
    Cu = linalg.cholesky(Sigma, lower=False)
    return {'mu': mu, 'Cu': Cu, 'Sigma': Sigma, 'maxlp': logpost(mu)}

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
    beta is exactly N(mu, A A^T).  F is the importance weight p(beta)/q(beta).
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

"""Chopin-Gerber's Pima experiment, with the spectral estimator alongside.

Protocol theirs (Section 5.2): 50 independent runs per configuration, no exact
value available, so the reported quantity is the Relative Variance
     rel-var = Var(est) / mean(est)^2 .
"""
import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')


def pima_sweep(d, ns, nreps, orders, theta, m_max, variant, scale=1.5, seed=0):
    rng = np.random.default_rng(seed)
    F, L = gaussian_integrand(d, scale_prop=scale)
    phi = cube_integrand(d, variant=variant)
    rows, cache = [], {}

    for n in ns:
        m, M, K = budget_split(int(n), d, theta=theta, m_max=m_max)
        key = (m, M)
        if key not in cache:
            cache[key] = quad_coefficients(F, d, m, M)
        est = []
        for _ in range(nreps):
            o = hermite_estimate(F, d, m, M, K, rng=rng, coeffs=cache[key])
            est.append(o['est']); nev = o['nevals']
        e = np.array(est)
        rows.append({'method': 'spectral', 'order': np.nan, 'nevals': nev,
                     'mean': e.mean(), 'relvar': e.var(ddof=1) / e.mean() ** 2,
                     'm': m, 'M': M, 'K': K})
        print('  spectral n=%-9d mean=%.6e rel-var=%.3e (m=%d,M=%d,K=%d)'
              % (nev, e.mean(), rows[-1]['relvar'], m, M, K))

    from cubic_strat import core as cg_core
    for r in orders:
        for n in ns:
            k = max(2, int(round((n / r) ** (1.0 / d))))
            est, nev = [], None
            try:
                for _ in range(nreps):
                    out = cg_core.vanish_estimates(k, d, order=r, phi=phi)
                    est.append(float(out['estimates'][r - 1]))
                    nev = int(out['nevals'][r - 1])
            except Exception as exc:
                print('   [skip] CG r=%d k=%d: %s' % (r, k, exc)); continue
            e = np.array(est)
            rows.append({'method': 'CG vanishing', 'order': r, 'nevals': nev,
                         'mean': e.mean(), 'relvar': e.var(ddof=1) / e.mean() ** 2})
            print('  CG r=%-2d  n=%-9d mean=%.6e rel-var=%.3e'
                  % (r, nev, e.mean(), rows[-1]['relvar']))
    return pd.DataFrame(rows)


def plot_pima(df, d, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cmap = plt.get_cmap('tab10')
    for i, (r, g) in enumerate(df[df.method == 'CG vanishing'].groupby('order')):
        g = g.sort_values('nevals')
        ax.loglog(g['nevals'], g['relvar'], lw=1.4, marker='o', ms=3,
                  color=cmap(i), label='CG vanishing, $r=%d$' % r)
    g = df[df.method == 'spectral'].sort_values('nevals')
    ax.loglog(g['nevals'], g['relvar'], lw=2.6, color='k', marker='s', ms=4,
              label='spectral estimator')
    ax.set_xlabel('nr evaluations'); ax.set_ylabel('rel-var')
    ax.set_title('Pima $s=%d$' % d)
    ax.grid(True, which='both', alpha=0.25); ax.legend(fontsize=7.5)
    fig.tight_layout()
    for e in ('pdf', 'png'): fig.savefig(path + '.' + e)


def run_pima(d, nmin, nmax, npts, nreps, orders, theta=0.5, m_max=400,
             variant='prop1', scale=1.5, outdir='pima_results'):
    os.makedirs(outdir, exist_ok=True)
    ns = np.unique(np.round(np.exp(np.linspace(np.log(nmin), np.log(nmax), npts)))).astype(int)
    print('=== Pima s=%d, lambda=%.1f ===' % (d, scale))
    df = pima_sweep(d, ns, nreps, orders, theta, m_max, variant, scale=scale)
    df['d'] = d; df['lam'] = scale
    df.to_csv(os.path.join(outdir, 'pima%d_lam%.1f.csv' % (d, scale)), index=False)
    plot_pima(df, d, os.path.join(outdir, 'pima%d_lam%.1f' % (d, scale)))
    return df

dfp2 = run_pima(d=2, nmin=200, nmax=50000, npts=6, nreps=50,
                orders=[1, 2, 4, 6], m_max=300)
dfp4a = run_pima(d=4, nmin=1000, nmax=40000, npts=5, nreps=50,
                 orders=[1, 2, 4], m_max=60, scale=1.5)
dfp4b = run_pima(d=4, nmin=1000, nmax=40000, npts=5, nreps=50,
                 orders=[1, 2, 4], m_max=60, scale=3.0)

from IPython.display import Image, display
import pandas as pd, glob

for f in sorted(glob.glob('pima_results/*.png')):
    print(f); display(Image(f))

for f in sorted(glob.glob('pima_results/*.csv')):
    df = pd.read_csv(f)
    lab = df.apply(lambda r: 'spectral' if r['method'] == 'spectral'
                   else 'CG r=%d' % r['order'], axis=1)
    print('\n===', f, '===')
    display(df.assign(row=lab).pivot_table(index='row', columns='nevals',
                                           values='relvar').round(12))
    print('means:', df.groupby(lab)['mean'].mean().round(6).to_dict())

