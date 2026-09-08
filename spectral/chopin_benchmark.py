"""
Chopin-Gerber's own benchmark, with the Hermite estimator dropped in.

Their test problems are integrals over [0,1]^s:

        I(f) = int_{[0,1]^s} f(u) du.

The Hermite estimator is defined against the Gaussian measure, so it attacks the
identical integral through the probability integral transform  u = Phi(z).

        I(f) = E[ F(Z) ],   F(z) = f(Phi(z)),   Z ~ N(0, I_s).

Nothing about the problem changes. Same f, same true value, same target, only
which measure the estimator integrates against.

Protocol is theirs (Section 5): nreps = 50 independent runs per configuration,
x-axis = number of evaluations of f, y-axis = rel-mse = mean[(est/I(f) - 1)^2],
log-log.  Their estimator is called through their own code
(`cubic_strat.core.estimate`) at orders r = 1, 2, 4, 6.

Usage
-----
python chopin_benchmark.py --d 1
python chopin_benchmark.py --d 2 --nreps 50
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import factorial
from scipy.stats import norm

warnings.filterwarnings('ignore')



# Chopin-Gerber's test problems, verbatim from nonvanish_xp/dick*D.py

def problem(d):
    if d == 1:
        return dict(
            ident='dick1D', d=1, true_val=1.0,
            title=r'$f_1(u)=ue^u$',
            phi=lambda u: (u * np.exp(u)).reshape(u.shape[0], -1).sum(axis=1))
    if d == 2:
        return dict(
            ident='dick2D', d=2, true_val=np.exp(1.) - 2.,
            title=r'$f_2(u)=u_2e^{u_1u_2}$',
            phi=lambda u: u[:, 1] * np.exp(np.prod(u, axis=-1)))
    if d == 3:
        return dict(
            ident='dick3D', d=3, true_val=np.exp(1.) - 2.5,
            title=r'$f_3(u)=u_2u_3^2e^{u_1u_2u_3}$',
            phi=lambda u: u[:, 1] * u[:, 2] ** 2 * np.exp(np.prod(u, axis=-1)))
    if d == 4:
        return dict(
            ident='dick4D', d=4, true_val=np.exp(1.) - 8. / 3.,
            title=r'$f_4(u)=u_2u_3^2u_4^3e^{u_1u_2u_3u_4}$',
            phi=lambda u: (u[:, 1] * u[:, 2] ** 2 * u[:, 3] ** 3
                           * np.exp(np.prod(u, axis=-1))))
    if d == 6:
        return dict(
            ident='dick6D', d=6,
            true_val=np.exp(1.) - np.sum(1. / factorial(np.arange(6))),
            title=r'$f_6(u)=(\prod_{i=2}^6 u_i^{i-1})e^{\prod_i u_i}$',
            phi=lambda u: np.prod([u[:, i] ** i for i in range(1, 6)], axis=0)
            * np.exp(np.prod(u, axis=-1)))
    raise ValueError(d)


def gaussian_pullback(pb):
    """F(z) = f(Phi(z)); same integral, Gaussian measure."""
    f = pb['phi']

    def F(z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            z = z[:, None]
        return np.asarray(f(norm.cdf(z))).ravel()
    return F


# the two estimators

def cg_run(pb, n, r):
    from cubic_strat import core as cg_core
    """Their estimator, their code.  n = min(r,3) * k^d."""
    per = min(r, 3)
    k = max(int(round((n / per) ** (1.0 / pb['d']))),
            2 if r <= 2 else 3 * r // 2 - 1)
    est = cg_core.estimate(k, pb['d'], order=r, phi=pb['phi'])
    return float(est), per * k ** pb['d']


def hermite_run(F, d, n, theta, m_max, rng, cache):
    """The Hermite estimator.  n = m^d + 2K."""
    m, M, K = budget_split(int(n), d, theta=theta, m_max=m_max)
    key = (m, M)
    if key not in cache:
        cache[key] = quad_coefficients(F, d, m, M)
    out = hermite_estimate(F, d, m, M, K, antithetic=True, rng=rng,
                              coeffs=cache[key])
    return out['est'], out['nevals'], m, M, K




def sweep_bench(pb, ns, nreps, orders, theta, m_max, seed=0):
    rng = np.random.default_rng(seed)
    F = gaussian_pullback(pb)
    truth = pb['true_val']
    rows, cache = [], {}

    for r in orders:
        for n in ns:
            try:
                ests, nev = [], None
                for _ in range(nreps):
                    e, nev = cg_run(pb, n, r)
                    ests.append(e)
            except Exception as exc:
                if 'TooSmallk' not in type(exc).__name__:
                    print('   [skip] CG r=%d: %s' % (r, exc))
                    break
                continue
            except NotImplementedError as exc:
                print('   [skip] CG r=%d: %s' % (r, exc))
                break
            e = np.asarray(ests)
            rows.append({'method': 'CG r=%d' % r, 'family': 'CG', 'r': r,
                         'n': nev, 'rel_mse': float(np.mean((e / truth - 1.) ** 2)),
                         'rmse': float(np.sqrt(np.mean((e - truth) ** 2)))})
            print('   CG r=%-2d n=%-9d rel-mse=%.4e' % (r, nev, rows[-1]['rel_mse']))

    for n in ns:
        ests, nev, info = [], None, None
        for _ in range(nreps):
            e, nev, m, M, K = hermite_run(F, pb['d'], n, theta, m_max, rng, cache)
            ests.append(e)
            info = (m, M, K)
        e = np.asarray(ests)
        rows.append({'method': 'Hermite', 'family': 'Hermite', 'r': np.nan,
                     'n': nev, 'rel_mse': float(np.mean((e / truth - 1.) ** 2)),
                     'rmse': float(np.sqrt(np.mean((e - truth) ** 2))),
                     'm': info[0], 'M': info[1], 'K': info[2]})
        print('   Hermite  n=%-9d rel-mse=%.4e   (m=%d, M=%d, K=%d)'
              % (nev, rows[-1]['rel_mse'], *info))

    return pd.DataFrame(rows)


def fit_slopes_bench(df, floor=1e-30):
    out = []
    for meth, g in df.groupby('method'):
        g = g[g['rel_mse'] > floor].sort_values('n')
        if len(g) < 3:
            out.append({'method': meth, 'slope_relmse': np.nan,
                        'npts': len(g), 'note': 'at machine precision'})
            continue
        b1, b0 = np.polyfit(np.log(g['n']), np.log(g['rel_mse']), 1)
        out.append({'method': meth, 'slope_relmse': b1,
                    'slope_rmse': b1 / 2, 'npts': len(g), 'note': ''})
    return pd.DataFrame(out)


def plot_bench(df, slopes, pb, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cmap = plt.get_cmap('tab10')
    for i, (meth, g) in enumerate(df.groupby('method')):
        g = g.sort_values('n')
        sl = slopes.loc[slopes['method'] == meth, 'slope_relmse']
        sl = sl.values[0] if len(sl) else np.nan
        lab = meth if np.isnan(sl) else '%s  [%.2f]' % (meth, sl)
        style = dict(lw=2.2, marker='') if meth == 'Hermite' else dict(lw=1.2,
                                                                      marker='o',
                                                                      ms=3)
        ax.loglog(g['n'], np.maximum(g['rel_mse'], 1e-33),
                  color='k' if meth == 'Hermite' else cmap(i),
                  label=lab, **style)
    ax.set_xlabel('nr evaluations')
    ax.set_ylabel('rel-mse')
    ax.set_title('%s   (s = %d)' % (pb['title'], pb['d']))
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_benchmark(d, orders, nmin, nmax, npts, nreps, theta, m_max, outdir='chopin_bench'):
    import os
    os.makedirs(outdir, exist_ok=True)
    pb = problem(d)
    ns = np.unique(np.round(np.exp(np.linspace(np.log(nmin), np.log(nmax), npts)))).astype(int)
    print('=== %s, true value %.12f ===' % (pb['ident'], pb['true_val']))
    df = sweep_bench(pb, ns, nreps, orders, theta, m_max)
    sl = fit_slopes_bench(df)
    df.to_csv(os.path.join(outdir, '%s.csv' % pb['ident']), index=False)
    sl.to_csv(os.path.join(outdir, '%s_slopes.csv' % pb['ident']), index=False)
    plot_bench(df, sl, pb, os.path.join(outdir, '%s.png' % pb['ident']))
    sl = sl.copy()
    sl['CG_predicted'] = [(-1. - 2. * int(m.split('=')[1]) / d) if m.startswith('CG') else np.nan
                          for m in sl['method']]
    print(sl.to_string(index=False))
    return df, sl
