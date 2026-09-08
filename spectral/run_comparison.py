"""
Hermite spectral control-variate estimator  vs  Chopin-Gerber

For a common grid of evaluation budgets n, every method is run `nreps` times on
the same Gaussian integration problem; we record the empirical RMSE against the
exact value and fit  log RMSE = b0 + b1 log n  by OLS

Methods
-------
mc            crude Monte Carlo, n draws                          (slope -1/2)
hermite       antithetic spectral estimator, m,K grown together   (this thesis)
hermite_plain non-antithetic version
hermite_fm<m> antithetic with m FIXED, only K grows               (diagnostic)
cg<r>         CG non-vanishing estimator (their eq. 9) applied to
              g(u) = f(Phi^{-1}(u)) on [0,1]^s,  n = 3k^s
cgv<r>        CG vanishing estimator (their eq. 16) applied to CG's own
              transform of  G(x) = f(x) phi(x)

Usage
-----
python run_comparison.py --s 1 --nreps 200 --nmax 20000
python run_comparison.py --s 2 --nreps 200 --nmax 200000 --targets relu3 abs invquad
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')




# single realisations


def run_mc(target, s, n, rng):
    Z = rng.standard_normal((n, s))
    return float(np.mean(target(Z))), n


def _cg():
    from cubic_strat import core as cg_core
    return cg_core


def run_cg_invcdf(target, s, n, r, rng):
    cg_core = _cg()
    """CG non-vanishing estimator on g = f o Phi^{-1}.  n = 3 k^s (r>=3)."""
    per = min(r, 3)
    k = int(round((n / per) ** (1.0 / s)))
    kmin = 2 if r <= 2 else (3 * r // 2 - 1)
    k = max(k, kmin)
    phi = invcdf_phi(target)
    est = cg_core.estimate(k, s, order=r, phi=phi)
    return float(est), per * k ** s


def run_cg_vanish(target, s, n, r, rng):
    cg_core = _cg()
    """CG vanishing estimator on CG's own R^s -> [0,1]^s transform."""
    k = max(2, int(round((n / r) ** (1.0 / s))))
    phi = vanishing_phi(target)
    out = cg_core.vanish_estimates(k, s, order=r, phi=phi)
    return float(out['estimates'][r - 1]), int(out['nevals'][r - 1])


# sweep

def budget_grid(nmin, nmax, npts):
    return np.unique(np.round(np.exp(np.linspace(np.log(nmin), np.log(nmax),
                                                 npts)))).astype(int)


def sweep(target, s, ns, nreps, methods, theta=0.5, m_max=800, fixed_m=None,
          seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for meth in methods:
        for n in ns:
            ests, nev = [], None
            try:
                if meth == 'mc':
                    for _ in range(nreps):
                        e, nev = run_mc(target, s, int(n), rng)
                        ests.append(e)

                elif meth.startswith('hermite'):
                    anti = not meth.endswith('plain')
                    if meth.startswith('hermite_fm'):
                        m = int(meth[len('hermite_fm'):])
                        M = m
                        K = max(1, (int(n) - m ** s) // 2)
                        if K < 5:
                            continue
                    else:
                        m, M, K = budget_split(int(n), s, theta=theta,
                                                  antithetic=anti, m_max=m_max)
                    b = quad_coefficients(target, s, m, M)
                    for _ in range(nreps):
                        out = hermite_estimate(target, s, m, M, K,
                                                  antithetic=anti, rng=rng,
                                                  coeffs=b)
                        ests.append(out['est'])
                        nev = out['nevals']

                elif meth.startswith('cgv'):
                    r = int(meth[3:])
                    for _ in range(nreps):
                        e, nev = run_cg_vanish(target, s, int(n), r, rng)
                        ests.append(e)

                elif meth.startswith('cg'):
                    r = int(meth[2:])
                    for _ in range(nreps):
                        e, nev = run_cg_invcdf(target, s, int(n), r, rng)
                        ests.append(e)
                else:
                    raise ValueError(meth)
            except Exception as exc:
                if 'TooSmallk' not in type(exc).__name__:
                    print('   [skip] %s n=%d: %s' % (meth, n, exc))
                continue

            e = np.asarray(ests)
            rows.append({'method': meth, 'n_target': int(n), 'n': int(nev),
                         'rmse': float(np.sqrt(np.mean((e - target.truth) ** 2))),
                         'bias': float(e.mean() - target.truth),
                         'sd': float(e.std(ddof=1)) if len(e) > 1 else np.nan,
                         'nreps': len(e)})
            print('   %-14s n=%-9d rmse=%.4e' % (meth, nev, rows[-1]['rmse']))
    return pd.DataFrame(rows)


def fit_slopes(df, target, rel_floor=1e-12, min_points=3):
    """OLS slope of log rmse on log n, discarding points at the precision floor."""
    out = []
    for meth, g in df.groupby('method'):
        g = g[g['rmse'] > rel_floor * max(abs(target.truth), 1.0)]
        g = g.sort_values('n')
        if len(g) < min_points:
            out.append({'method': meth, 'slope': np.nan, 'npts': len(g),
                        'note': 'hit machine precision'})
            continue
        x, y = np.log(g['n'].values), np.log(g['rmse'].values)
        b1, b0 = np.polyfit(x, y, 1)
        out.append({'method': meth, 'slope': b1, 'intercept': b0,
                    'npts': len(g), 'note': ''})
    return pd.DataFrame(out)


# plotting

def predicted_slope(method, s, target):
    """Theoretical exponent each method is supposed to hit on this target."""
    r = target.smooth
    if method == 'mc':
        return -0.5
    if method.startswith('hermite_fm'):
        return -0.5                       # M frozen: no approximation gain
    if method.startswith('hermite'):
        return np.nan if not np.isfinite(r) else -0.5 - r / (2.0 * s)
    if method.startswith('cgv') or method.startswith('cg'):
        rr = int(method.lstrip('cgv'))
        reff = rr if not np.isfinite(r) else min(rr, r)
        return -0.5 - reff / s
    return np.nan


def make_plot(df, slopes, target, s, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for meth, g in df.groupby('method'):
        g = g.sort_values('n')
        sl = slopes.loc[slopes['method'] == meth, 'slope']
        sl = sl.values[0] if len(sl) else np.nan
        lab = meth if np.isnan(sl) else '%s (slope %.2f)' % (meth, sl)
        ax.loglog(g['n'], g['rmse'], marker='o', ms=3.5, lw=1.3, label=lab)

    nn = np.array(sorted(df['n'].unique()), dtype=float)
    ref = df['rmse'].max()
    for expo, sty in [(-0.5, ':'), (-0.5 - 1.0 / (2 * s), '--')]:
        ax.loglog(nn, ref * (nn / nn[0]) ** expo, sty, color='0.5', lw=1.0,
                  label=r'$n^{%.2f}$' % expo)

    ax.set_xlabel('n (evaluations of f)')
    ax.set_ylabel('RMSE')
    ax.set_title('s = %d,  %s' % (s, target.label))
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)




FAMILIES = {'standard': make_targets, 'relu': relu_family,
            'tail': tail_family, 'nonsep': nonseparable_family}


def experiment(s, family, targets_wanted, methods, nmin, nmax, npts, nreps,
               theta, m_max, outdir, seed=0, plot_each=True):
    """One sweep: returns (rmse dataframe, slopes dataframe) and writes CSVs."""
    import os
    os.makedirs(outdir, exist_ok=True)
    T = FAMILIES[family](s)
    names = targets_wanted or list(T)
    ns = budget_grid(nmin, nmax, npts)
    all_df, all_sl = [], []
    for name in names:
        t = T[name]
        print('\n=== s=%d  %s  (truth %.12f) ===' % (s, name, t.truth))
        df = sweep(t, s, ns, nreps, methods, theta=theta, m_max=m_max, seed=seed)
        sl = fit_slopes(df, t)
        sl['predicted'] = [predicted_slope(m, s, t) for m in sl['method']]
        df['target'] = name; sl['target'] = name; sl['s'] = s
        if plot_each:
            make_plot(df, sl, t, s, os.path.join(outdir, 'rmse_s%d_%s.png' % (s, name)))
        all_df.append(df); all_sl.append(sl)
        print(sl.to_string(index=False))
    D = pd.concat(all_df); S = pd.concat(all_sl)
    D.to_csv(os.path.join(outdir, 'rmse_s%d.csv' % s), index=False)
    S.to_csv(os.path.join(outdir, 'slopes_s%d.csv' % s), index=False)
    return D, S
