# cubic_strat #
Stochastic integration with higher-order accuracy.

> **This is a fork.** The upstream package, by Nicolas Chopin and Mathieu
> Gerber, is unchanged and documented below. Added in this fork is `spectral/`,
> which implements a Hermite spectral estimator for Gaussian integrals and
> compares it against the two upstream estimators. See
> [`spectral/README.md`](spectral/README.md) for that work; everything else in
> this repository is upstream and untouched.

## Motivation ## 
This Python package implements the two estimators proposed in the following paper: 
[Higher-order stochastic integration through cubic stratification](https://arxiv.org/abs/2210.01554)
for estimating the integral $\int_{[0,1]^s} f(u)du$ of a function $f$. 
The package also contains script to reproduce the numerical experiments found
in the paper. 

## Non-vanishing estimator ## 
Consider the function $f(u)=\exp\{u_1 u_2^2\}$ over $[0, 1]^2$. To estimate its integral, you
must first define a Python function that computes $f$ for an array of vectors
in $[0, 1]^2$:
```python
    import numpy as np
    def f(u):
        return np.exp(u[:, 1] * u[:, 1]**2)
```
Then you may compute the estimator defined as $\widehat{I}_{r,k}(f)$ in the
paper as follows:
```python
    import cubic_strat as cubs
    k = 10
    r = 4  # order
    est = cubs.estimate(k, 2, order=r, phi=f)
```

## Vanishing estimator ##
It works the same way, except that: 
* you need to make sure that your function is indeed vanishing (i.e. it may be
  extended to a function over $\mathbb{R}^s$ which is zero outside 
  of $[0, 1]^s$, while still being $r-$times continuously differentiable.)
* The function below compute the vanishing estimators at all orders up to the
  given value, as explained in Section 4.2 in the paper:
```python
    # compute vanishing estimator at orders 1 to 10
    est = cubs.vanishing_estimates(k, 2, order=10, phi=f)
```

## Numerical experiments ##
The scripts that implement the numerical experiments in the paper may found in
the following two folders: vanishing_xp, and nonvanishing_xp. 

## TODO ##
* Remove deprecated parts (module numdiff, etc)

## Questions ##
Feel free to email me: nicolas.chopin@ensae.fr

---

## Additions in this fork ##

Estimators for Gaussian integrals $\int_{\mathbb{R}^s} f(x)\,\phi_{\mu,\Sigma}(x)\,dx$,
developed for an MSc dissertation, together with the code that compares them
against the upstream estimators on the upstream test problems.

| | |
|---|---|
| [`spectral/`](spectral/) | the estimators, the experiments, and their own README |
| [`reruns.ipynb`](reruns.ipynb) | a notebook that reproduces every figure and table from scratch |

The comparison calls the upstream estimators through this package rather than
reimplementing them, and reproduces the exponents of Theorem 1 of the paper
before drawing any comparison. One correction to the upstream code was needed
and is documented in `spectral/README.md`.
