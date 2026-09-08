# Spectral estimators for Gaussian integrals

Code for an MSc dissertation on unbiased estimation of

$$I_\phi(f) \;=\; \int_{\mathbb{R}^s} f(x)\,\phi_{\mu,\Sigma}(x)\,dx \;=\; \mathbb{E}_{X\sim N(\mu,\Sigma)}[f(X)],$$

and its comparison against the cubic-stratification estimators of Chopin and
Gerber (2022), which are called through the upstream package in the parent
directory rather than reimplemented here.

Two estimators are provided.

**The stratified estimator** (`stratified.py`) partitions $\mathbb{R}^s$ into
$k^s$ boxes of equal Gaussian probability using the normal quantiles, and on
each box averages an antithetic pair with even-order control variates
subtracted. It is the direct Gaussian analogue of the upstream construction. Its
rate is capped at $O(n^{-1/2-1/(2s)})$ regardless of the smoothness order,
because the outermost strata have widths that shrink only at the Mills-ratio
rate; this is what motivates the second estimator.

**The spectral estimator** (`hermite_cv.py`) instead expands $f$ in the Hermite
polynomials orthogonal to the Gaussian measure, approximates the coefficients by
Gauss–Hermite quadrature, and subtracts the truncated expansion as a control
variate. It is unbiased for every truncation degree $M$ and every quadrature
order $m$, and has no order parameter to select.

## Installation

```bash
pip install findiff particles
pip install -e ..          # the upstream package, from this directory
```

`findiff` supplies the numerical derivatives used by the upstream estimator;
`particles` supplies the Pima dataset.

## Layout

| file | contents |
|---|---|
| `hermite_cv.py` | the spectral estimator: orthonormal Hermite basis, Gauss–Hermite rule, tensorised coefficients, antithetic and plain variants, exact variance decomposition |
| `stratified.py` | the stratified estimator, its truncated-normal moments, and its figure |
| `gaussian_targets.py` | test integrands with exact values, including the smoothness ladder |
| `run_comparison.py` | budget sweeps, slope fitting, plots |
| `chopin_benchmark.py` | the upstream test functions, both estimators, upstream protocol |
| `pima_setup.py`, `integrands.py`, `run_pima.py` | the Pima marginal-likelihood experiment |
| `make_thesis_figures.py` | the dissertation figures |
| `results/` | the CSVs and figures behind the reported tables |

The notebook `../reruns.ipynb` reproduces everything without importing these
files; it inlines them so it can be run in a fresh environment.

## Reproducing the results

**The smoothness ladder** — the experiment from which the convergence rate is
measured. The family $f_p(z) = \sum_i (z_i - 0.3)_+^{\,p}$ has exactly $p$
square-integrable derivatives, since the $p$-th is a bounded step and the
$(p+1)$-th a Dirac mass, so varying $p$ varies the smoothness and nothing else.

```bash
python run_comparison.py --s 1 --family relu --nmin 400 --nmax 12000 \
    --npts 8 --nreps 200 --theta 0.25 --m_max 3000 \
    --methods hermite hermite_fm12 --outdir ladder_s1
python run_comparison.py --s 2 --family relu --nmin 1000 --nmax 200000 \
    --npts 8 --nreps 200 --theta 0.5 --m_max 400 \
    --methods hermite --outdir ladder_s2
```

Adding `cgv1 cgv2 cgv4` to `--methods` runs the upstream vanishing estimator on
the same integrands, which is the only setting in which the two exponents can be
compared directly.

**The upstream benchmark** — their functions $ue^u$, $u_2e^{u_1u_2}$ and
$u_2u_3^2e^{u_1u_2u_3}$, taken from `nonvanish_xp/dick*D.py`, attacked by both
estimators through the probability integral transform $u = \Phi(z)$.

```bash
python chopin_benchmark.py --d 1 --orders 1 2 4 6 --nreps 50
python chopin_benchmark.py --d 2 --orders 1 2 4 6 --nreps 50
python chopin_benchmark.py --d 3 --orders 1 2 4 6 --nreps 50
```

**The Pima marginal likelihood** — Bayesian logistic regression on the Pima
dataset, the problem of Section 5.2 of the paper.

```bash
cd .  # from spectral/
python run_pima.py --d 2 --nreps 50 --orders 1 2 4 6
python run_pima.py --d 4 --nreps 50 --orders 1 2 4
```

**The stratified estimator's figure**

```bash
python stratified.py
```

## Two departures from the upstream code

Both concern the Pima experiment only, and both were necessary for the two
estimators to be attacking the same integral.

**The log-Jacobian sign in `vanish_xp/pima_common.py`.** That routine forms
`lq = ljac - cst` and returns `exp(lp - lq - maxlp)`, so the log-Jacobian enters
negatively, whereas Proposition 1 of the paper requires
$f_{g,\psi}(u) = g(\psi_s(u)) \prod_i \psi_1'(u_i)$, that is positively; the
`psi` routine does return $dz/du$. The additive constant also carries a factor
$\tfrac12$ relative to $\log|\det C|$. As written the routine returns
$1.08\times10^{-4}$ at $s=2$ and $1.46\times10^{-8}$ at $s=4$, against correct
values of order $7.5\times10^{-2}$ and $2.2\times10^{-2}$. The published results
are unaffected, since the relative variance they report is invariant to the scale
of the integrand. `integrands.py` restores the sign and the constant, and offers
the original as `variant='as_written'` for comparison.

**The Laplace covariance.** `vanish_xp/prep_pima.py` takes $\Sigma$ from the
optimiser's `hess_inv`, which is BFGS's accumulated approximation rather than the
inverse Hessian; at $s=4$ the two differ by factors between $0.4$ and $7.5$ along
the diagonal, which is enough to make any importance-weighted estimator built on
it behave badly. `pima_setup.py` forms the Hessian in closed form,
$\sum_i \sigma_i(1-\sigma_i)x_ix_i^\top + \sigma_{\text{prior}}^{-2}I$, and does
not read `results/pima_mean_cov.pkl`.

## Notes on the implementation

Everything is written in the orthonormal basis
$\hat h_\alpha(z) = \prod_i He_{\alpha_i}(z_i)/\sqrt{\alpha_i!}$, so that
$\beta_\alpha = b_\alpha/\sqrt{\alpha!}$ and the variance identities lose their
factorial bookkeeping. $He_n(z)$ overflows for moderate $n$; the normalised
version does not.

Two Gauss–Hermite pitfalls are handled in `hermite_cv.py`.
`numpy.polynomial.hermite_e.hermegauss` overflows past $m \approx 150$. And the
textbook Golub–Welsch weights $w_q = V_{0q}^2$ have no relative accuracy for the
tiny outer weights, bottoming out near $10^{-60}$ where the true value is
$10^{-350}$, which destroys any expression multiplying a weight by something
exponentially large. Nodes are taken from the Jacobi eigenvalues and weights from
the Christoffel formula $w_q = 1/\sum_{n<m}\hat h_n(x_q)^2$, evaluated in a scaled
basis where every quantity is $O(1)$. This is accurate to $m \approx 4000$.

The cost of `quad_coefficients` is an $m^s$ grid and an $(m,M)$ matrix, and of
`eval_expansion` is $K M^s$; keep $m^s \lesssim 10^5$. In $s=1$ use a small
`--theta` to keep $m$ within `--m_max`.

`run_comparison.py` writes `rmse_s{s}.csv` and `slopes_s{s}.csv` into
`--outdir`, so two runs at the same dimension into the same directory will
overwrite one another. Use a distinct `--outdir` per run.
