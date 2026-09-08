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
