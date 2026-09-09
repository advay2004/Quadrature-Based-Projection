# The spectral estimator

Hermite control variates with coefficients from a tensorised Gauss--Hermite rule.
Produces Figures 2--5 and Tables 2--7 of the dissertation.

## The estimator

Fix a truncation degree $M$ and a quadrature order $m$. Form the coefficients

$$\hat\beta_\alpha = \frac{1}{\alpha!}\sum_{q=1}^{m^s} w_q\, f(\xi_q) H_\alpha(\xi_q)$$

once from the $m^s$ tensorised Gauss--Hermite nodes, then draw
$Z_1,\dots,Z_K \sim \mathcal{N}(0, I_s)$ independently of them and average

$$\hat I^{(A)}_M(f) = \frac{1}{K}\sum_{i=1}^{K}\left\{
\frac{f(Z_i)+f(-Z_i)}{2} - \sum_{\substack{0<|\alpha|<M \\ |\alpha| \text{ even}}}
\hat\beta_\alpha H_\alpha(Z_i)\right\}.$$

Since $\mathbb{E}[H_\alpha(Z)] = 0$ for every $|\alpha| > 0$, this is unbiased at
every $M$ and every $m$ — the coefficients are fixed before the sample is drawn, so
their error affects the variance and not the expectation. The parity relation
$H_\alpha(-z) = (-1)^{|\alpha|}H_\alpha(z)$ removes every odd-degree coefficient
identically, so only even $|\alpha|$ are retained.

The budget is $n = m^s + 2K$, split by a parameter $\theta \in (0,1)$ giving the
fraction spent on the coefficients:

$$m = \lfloor (\theta n)^{1/s} \rfloor, \qquad M = m, \qquad
K = \lceil (1-\theta)n/2 \rceil .$$

$M = m$ is forced from both sides: larger breaks the exactness of the rule on the
products $f H_\alpha$, and smaller wastes coefficients already paid for.

## Contents

| | |
|---|---|
| `Spectral Code.ipynb` | every run below, in order |
| `figs/` | the figures and CSVs the notebook writes |

## Reproducing the results

Open `Spectral Code.ipynb`, run the setup cells at the top, then any block
independently.

| Output | Call | Notes |
|---|---|---|
| Figure 3, Table 3 | ladder sweeps at $s=1$ and $s=2$, then `build_figures(...)` | writes `figs/fig_ladder.png` |
| Figure 4 | same call | writes `figs/fig_slope_vs_r.png` |
| Figure 2, Table 2 | `run_benchmark(...)` at $d=1,2,3$ | writes `chopin_bench/dick{1,2,3}D.png` |
| Table 4 | head-to-head sweep at $s=1$ | both estimators on the same integrands |
| Figure 5, Tables 6--7 | `run_pima(d=2, ...)` and `run_pima(d=4, ...)` | writes `pima_results/Pima{2,4}.png` |

The $s=2$ ladder and the $s=3$ benchmark are the long runs. Reduce `nreps` for a
quick look.

## Test problems

**Smoothness ladder.** $f_p(z) = \sum_i (z_i - c)_+^p$ with $c = 0.3$ and
$p = 1,\dots,5$. Differentiating $p$ times leaves a step function and the
$(p+1)$-th derivative is a Dirac mass, so $f_p$ has exactly $p$ square-integrable
weak derivatives. Varying $p$ changes the smoothness and nothing else, which is
what makes the increment per derivative interpretable. The kink sits at $0.3$
rather than the origin because $z=0$ maps to $u=1/2$ under the probability integral
transform, which is a cell boundary of the cubic stratification whenever $k$ is
even, and a singularity on a cell boundary flatters the comparison method.

**Chopin--Gerber problems.** Their `dick1D`, `dick2D`, `dick3D` integrands, brought
onto the Gaussian measure by $u = \Phi(z)$ coordinatewise. These are $C^\infty$, so
no finite smoothness index exists and the experiment measures adaptivity rather
than a rate.

**Pima.** Marginal likelihood of a Bayesian logistic regression at $s=2$ and $s=4$.
The Hessian is formed in closed form at the posterior mode rather than read off the
optimiser's accumulated approximation, which at $s=4$ differ by factors between
0.4 and 7.5 along the diagonal.

## What the results show

The estimator leads at $s=1$ and $s=2$ on every problem tested, reaching the double
precision floor at budgets where the comparison method needs its highest order and
roughly twice the evaluations. By $s=3$ it loses the lead. The tensor grid costs
$m^s$ evaluations, so the truncation degree grows only as $n^{1/s}$, and at
$n = 40{,}000$ with $s=4$ that leaves eleven nodes per axis, which are too few to exploit the
smoothness the integrand has. A sparse grid construction is the direct extension.

On the finite-smoothness ladder the fitted slopes are steeper than the proved bound
throughout, and their increments per derivative are close to $0.5$ at $s=1$ and
$0.25$ at $s=2$, consistent with $n^{-1/2-r/2s}$, which is the proved rate with the
$1/(6s)$ removed.
