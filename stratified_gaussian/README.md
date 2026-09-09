# The stratified Gaussian estimator

Equal-probability strata at the standard normal quantiles, corrected by a local
Taylor expansion about each conditional mean. Produces Figure 1 and Table 1 of the
dissertation.

This is the direct analogue of the cubic-stratified construction under a Gaussian
measure. It is included because its failure is what motivates the spectral
estimator, not because it is recommended.

## The estimator

For $k \ge 2$ let $q_j = \Phi^{-1}(j/k)$ with $q_0 = -\infty$ and $q_k = +\infty$,
and let $J_j = (q_{j-1}, q_j]$. The strata $S_c = \prod_i J_{j_i}$ partition
$\mathbb{R}^s$, and the coordinates of $Z$ being independent, each carries
probability exactly $k^{-s}$.

An unbounded interval has no geometric centre, so the centre of each stratum is its
conditional mean, $\bar z_j = k\{\varphi(q_{j-1}) - \varphi(q_j)\}$. A draw is
obtained coordinatewise by inversion,
$(X^+_c)_i = \Phi^{-1}\{(j_i - 1 + V_i)/k\}$ with $V_i \sim U(0,1)$, and the
estimator averages an antithetic pair with even-order control variates subtracted.


Reflecting in $z$ about the conditional mean is what the cube construction does
about a cell centre, and it works there because a cell is symmetric about its
centre. A Gaussian stratum is not symmetric about its conditional mean — at $k=10$
the outer finite stratum has mean $-1.0446$ against an interval midpoint of
$-1.0616$ — so the reflected point's law is not $\gamma$ restricted to that stratum
and the estimator acquires a bias of order $10^{-3}$ at small $k$. Reflecting
$V \mapsto 1-V$ keeps the partner inside its own stratum with exactly the right
law, and unbiasedness is preserved.

The truncated-normal central moments $\nu^\alpha_c$ come from the two-term
recursion

$$\mu'_n = (n-1)\mu'_{n-2} + k\left\{q_{j-1}^{n-1}\varphi(q_{j-1}) -
q_j^{n-1}\varphi(q_j)\right\},$$

obtained by integration by parts, with the boundary terms vanishing at an infinite
endpoint. All of $\{\nu^\alpha_c : |\alpha| < r\}$ over every stratum needs only
the $kr$ numbers $\{\mu'_n\}$, at $O(kr)$ cost.

Derivatives are supplied in closed form rather than by finite differences, which
isolates the stratification from the differencing error. The budget is
$n = 2k^s$, one antithetic pair per stratum.

## Contents

| | |
|---|---|
| `Stratified.ipynb` | the estimator and the run that produces Figure 1 and Table 1 |
| `figs` | `fig_strat.png`|

## Reproducing Figure 1 and Table 1

Open `Stratified.ipynb`, run the cell defining the estimator, then

```python
make_figure(nreps=200)
```

Renders inline, writes `figs/fig_strat.{png,pdf}`, and prints the four fitted
slopes that make up Table 1. The $s=2$ panel takes a few minutes at `nreps=200`;
drop to 50 for a quick look.

Test integrands are $f(z) = \sin(z+1)$ at $s=1$ and
$f(z_1,z_2) = \sin(z_1+1) + \cos(z_2)$ at $s=2$. Both are $C^\infty$ with every
derivative bounded by 1, so the smoothness of the integrand places no restriction
on the order that can be tested, and the second is separable, so mixed partials
vanish and the correction stays in closed form.

## What the results show

The fitted slopes at $r=2$ and $r=4$ are close to indistinguishable at both
dimensions, and the predicted slope itself moves from $-1.00$ to $-0.75$ as $s$
goes from 1 to 2. Doubling the order of the local expansion buys no exponent.

This is the cap of

$$\mathrm{RMSE} = O\!\left(n^{-\frac12-\frac{1}{2s}}(\log n)^{-r/2}\right),$$

in which $r$ appears only inside the logarithm. The cause is that the extreme
quantiles satisfy $\Phi^{-1}(1/k) \sim -\sqrt{2\log k}$, so the displacement in an
end stratum shrinks only logarithmically in $k$, and since
$k^s - (k-2)^s = 2sk^{s-1} + O(k^{s-2})$ a growing number of strata have at least
one coordinate at an end. Every control variate here is local to one stratum, and
refining the local model cannot help when the difficulty is that the extreme strata
are unbounded. Control variates defined globally on $\mathbb{R}^s$ avoid this,
which is the spectral estimator.
