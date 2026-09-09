# RUN 1: smoothness ladder at $s=1$Feeds **Table 3** (upper half) and the **left panel of Figure 3**. The`hermite_fm12` method is the frozen-$M$ control curve drawn in that panel.
D1, S1 = experiment(
    s=1, family='relu', targets_wanted=None,
    methods=['hermite', 'hermite_fm12'],
    nmin=400, nmax=12000, npts=8, nreps=200,
    theta=0.25, m_max=3000, outdir='ladder_s1', seed=0)
S1[['target', 'method', 'slope', 'predicted']]

# RUN 2 : smoothness ladder at $s=2$Feeds **Table 3** (lower half) and the **right panel of Figure 3**.
D2, S2 = experiment(
    s=2, family='relu', targets_wanted=None,
    methods=['hermite'],
    nmin=1000, nmax=200000, npts=8, nreps=200,
    theta=0.5, m_max=400, outdir='ladder_s2', seed=0)
S2[['target', 'method', 'slope', 'predicted']]

import pandas as pd, re

rows = []
for s, path in ((1, 'ladder_s1/slopes_s1.csv'), (2, 'ladder_s2/slopes_s2.csv')):
    df = pd.read_csv(path)
    df = df[df.method == 'hermite'].copy()
    df['p'] = df.target.str.extract(r'(\d+)$').astype(int)
    df = df.sort_values('p')
    df['increment'] = -df.slope.diff()
    for _, r in df.iterrows():
        rows.append({'s': s, 'p': r.p, 'fitted': round(r.slope, 3),
                     'increment': None if pd.isna(r.increment) else round(r.increment, 3)})

T = pd.DataFrame(rows)
display(T)

# RUN 3: Parity check
Dp, Sp = experiment(
    s=1, family='relu', targets_wanted=None,
    methods=['hermite', 'hermite_plain'],
    nmin=400, nmax=12000, npts=8, nreps=200,
    theta=0.25, m_max=3000, outdir='ladder_s1_parity', seed=0)

import pandas as pd
d = pd.read_csv('ladder_s1_parity/slopes_s1.csv')
d['p'] = d.target.str.extract(r'(\d+)$').astype(int)
T = d.pivot_table(index='p', columns='method', values='slope').sort_index()
T['trend'] = [-0.5 - p/2 for p in T.index]
for m in ('hermite', 'hermite_plain'):
    T['inc_' + m] = -T[m].diff()
    T['off_' + m] = T['trend'] - T[m]
display(T.round(3))

# Figures 3 and 4Writes figs/fig_ladder.png and figs/fig_slope_vs_r.png. Requires RUN 1 and RUN 2 to have completed.

"""
Build the two ladder figures, which the per-target sweep does not produce
directly.

  Figure 3  fig_ladder.png      two panels (s=1, s=2), RMSE against n, one
                                curve per smoothness index p, plus the
                                frozen-M control curve at s=1
  Figure 4  fig_slope_vs_r.png  fitted exponent against p, with the lines
                                -1/2 - p/(2s) overlaid

Both read the CSVs written by RUN 1 and RUN 2. `load` accepts either a single
path or a list, so a sweep run in chunks can be concatenated; duplicate
(target, method, n) rows are dropped.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})


def load(paths):
    if isinstance(paths, str):
        paths = [paths]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if 'n' in df.columns:
        df = df.drop_duplicates(subset=['target', 'method', 'n'])
    else:
        df = df.drop_duplicates(subset=['target', 'method'])
    return df


def r_of(target):
    m = re.search(r'(\d+)$', str(target))
    return int(m.group(1)) if m else np.nan


# Figure 3 : the ladder


def fig_ladder(d1, d2, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    cmap = plt.get_cmap('viridis')

    for ax, df, s in ((axes[0], d1, 1), (axes[1], d2, 2)):
        if df is None or df.empty:
            ax.set_visible(False)
            continue
        h = df[df['method'] == 'hermite'].copy()
        h['r'] = h['target'].map(r_of)
        rs = sorted(h['r'].dropna().unique())

        for i, r in enumerate(rs):
            g = h[h['r'] == r].sort_values('n')
            ax.loglog(g['n'], g['rmse'], marker='o', ms=3.5, lw=1.4,
                      color=cmap(i / max(len(rs) - 1, 1)),
                      label=r'$p=%d$' % r)

        # frozen-M control, if present
        fm = df[df['method'].str.startswith('hermite_fm')]
        if not fm.empty:
            tgt = fm['target'].iloc[0]
            g = fm[fm['target'] == tgt].sort_values('n')
            ax.loglog(g['n'], g['rmse'], ls='--', lw=1.6, color='crimson',
                      label=r'$M$ frozen ($r=%d$)' % r_of(tgt))

        # reference slopes -1/2 - r/(2s) anchored at the r-th curve's first point
        for i, r in enumerate(rs):
            g = h[h['r'] == r].sort_values('n')
            nn = g['n'].values.astype(float)
            if len(nn) < 2:
                continue
            expo = -0.5 - r / (2.0 * s)
            ref = g['rmse'].values[0] * (nn / nn[0]) ** expo
            ax.loglog(nn, ref, ls=':', lw=0.9, color='0.55',
                      label=r'$n^{-1/2-p/2s}$' if i == 0 else None)

        ax.set_xlabel(r'$n$ (evaluations of $f$)')
        ax.set_ylabel('RMSE')
        ax.set_title(r'$s=%d$' % s)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(loc='lower left', ncol=2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


# Figure 4 : fitted exponent against p

def fig_slope_vs_r(s1, s2, path):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    marks = {1: 'o', 2: 's'}
    cols = {1: 'tab:blue', 2: 'tab:orange'}

    for s, df in ((1, s1), (2, s2)):
        if df is None or df.empty:
            continue
        g = df[df['method'] == 'hermite'].copy()
        g['r'] = g['target'].map(r_of)
        g = g.dropna(subset=['r', 'slope']).sort_values('r')
        ax.plot(g['r'], g['slope'], marker=marks[s], ms=6, lw=0, color=cols[s],
                label=r'fitted, $s=%d$' % s)

        rr = np.linspace(g['r'].min() - 0.2, g['r'].max() + 0.2, 50)
        ax.plot(rr, -0.5 - rr / (2.0 * s), ls='--', lw=1.3, color=cols[s],
                label=r'$-\frac{1}{2}-\frac{r}{2s}$, $s=%d$' % s)

    ax.set_xlabel(r'smoothness index $r$')
    ax.set_ylabel('fitted RMSE-vs-$n$ slope')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print('wrote', path)


def build_figures(s1_rmse, s2_rmse, s1_slopes, s2_slopes, outdir='figs'):
    os.makedirs(outdir, exist_ok=True)
    d1, d2 = load(s1_rmse), load(s2_rmse)
    t1, t2 = load(s1_slopes), load(s2_slopes)
    fig_ladder(d1, d2, os.path.join(outdir, 'fig_ladder.pdf'))
    fig_ladder(d1, d2, os.path.join(outdir, 'fig_ladder.png'))
    fig_slope_vs_r(t1, t2, os.path.join(outdir, 'fig_slope_vs_r.pdf'))
    fig_slope_vs_r(t1, t2, os.path.join(outdir, 'fig_slope_vs_r.png'))

build_figures('ladder_s1/rmse_s1.csv', 'ladder_s2/rmse_s2.csv',
              'ladder_s1/slopes_s1.csv', 'ladder_s2/slopes_s2.csv')
from IPython.display import Image, display
display(Image('figs/fig_ladder.png')); display(Image('figs/fig_slope_vs_p.png'))

#RUN 4 — Chopin--Gerber test problems Feeds Table 2 and Figure 2 (chopin_bench/dick1D.png, dick2D.png,dick3D.png).
"""
Chopin-Gerber's own benchmark, with the Hermite estimator dropped in.

Their test problems are integrals over [0,1]^s:

        I(f) = int_{[0,1]^s} f(u) du.

The Hermite estimator is defined against the Gaussian measure, so it attacks the
identical integral through the probability integral transform  u = Phi(z):

        I(f) = E[ F(Z) ],   F(z) := f(Phi(z)),   Z ~ N(0, I_s).

Nothing about the problem changes -- same f, same true value, same target -- only
which measure the estimator integrates against.

Protocol is theirs (Section 5): nreps = 50 independent runs per configuration,
x-axis = number of evaluations of f, y-axis = rel-mse = mean[(est/I(f) - 1)^2],
log-log.  Their estimator is called through their own code
(`cubic_strat.core.estimate`) at orders r = 1, 2, 4, 6.

Usage
-----
python chopin_benchmark.py --d = 1
python chopin_benchmark.py --d = 2 --nreps 50
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

df1, sl1 = run_benchmark(d=1, orders=[1, 2, 4, 6],
                         nmin=30, nmax=20000, npts=9, nreps=50,
                         theta=0.4, m_max=500)

df2, sl2 = run_benchmark(d=2, orders=[1, 2, 4, 6],
                         nmin=100, nmax=300000, npts=9, nreps=50,
                         theta=0.5, m_max=400)

df3, sl3 = run_benchmark(d=3, orders=[1, 2, 4, 6],
                        nmin=1000, nmax=200000, npts=8, nreps=50,
                        theta=0.5, m_max=200)

from IPython.display import Image, display
import numpy as np, pandas as pd

for d in (1, 2, 3):
    print('\n' + '=' * 40, 's = %d' % d)
    display(Image('chopin_bench/dick%dD.png' % d))
    sl = pd.read_csv('chopin_bench/dick%dD_slopes.csv' % d)
    display(sl)
    df = pd.read_csv('chopin_bench/dick%dD.csv' % d)
    g = df[df.method == 'Hermite'].sort_values('n')
    print('local slopes:',
          np.round(np.diff(np.log(g.rel_mse.values)) / np.diff(np.log(g.n.values)), 2))


# Run 5: head to head
import pandas as pd, numpy as np, matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
d = pd.read_csv('head2head_s1/slopes_s1.csv')
d['p'] = d.target.str.extract(r'(\d+)$').astype(int)
T = d.pivot_table(index='p', columns='method', values='slope').sort_index()
display(T.round(3))
print('\nmean gain per derivative:')
print((-T.diff().mean()).round(3))

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for m in T.columns:
    ax.plot(T.index, T[m], marker='o', ms=5, lw=1.6, label=m)
pp = np.array(T.index, float)
ax.plot(pp, -0.5 - pp/2, 'k--', lw=1.2, label=r'$-\frac{1}{2}-\frac{p}{2s}$')
ax.plot(pp, -0.5 - pp,   'k:',  lw=1.2, label=r'$-\frac{1}{2}-\frac{p}{s}$')
ax.set_xlabel('number of derivatives $p$')
ax.set_ylabel('fitted RMSE-vs-$n$ slope')
ax.set_title('$s=1$: both estimators on the same integrands')
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.savefig('figs/head2head_s1.pdf', bbox_inches='tight')
plt.show()


