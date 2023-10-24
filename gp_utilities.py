# Utility functions for GPs.
#import george
#from george import kernels
import celerite2
from celerite2 import terms
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pdb import set_trace

def neg_ln_like(par, gp, y):
    if par[-1] < 1e-2 or par[-1] > 100.:
        return -np.inf
    elif np.log10(np.exp(par[-2])) < -12 or np.log10(np.exp(par[-2])) > -4.:
        return -np.inf

    gp.set_parameter_vector(par)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(par, gp, y):
    gp.set_parameter_vector(par)
    return -gp.grad_log_likelihood(y)

def lnprob(par, gp, y):
    # Trivial uniform prior.
    if par[-1] < 1e-2 or par[-1] > 100.:
        return -np.inf
    elif np.log10(np.exp(par[-2])) < -12 or np.log10(np.exp(par[-2])) > -4.:
        return -np.inf
    elif (abs(par[:-2] > 10.)).any():
        return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(par)
    try:
        return gp.lnlikelihood(y, quiet=True)
    except AttributeError:
        return -np.inf

def generate_data(t, plots=True):

    yerr = stats.loguniform.rvs(5e-4, 5e-3)
    sigmaRot = stats.loguniform.rvs(1e-3, 1e-1)
    period = np.random.uniform(1.5, 100.)
    #period = np.rancdom.uniform(200., 2000.)
    Q0 = 1.
    dQ = Q0/np.random.uniform(1., 5.)
    term1 = terms.RotationTerm(sigma=sigmaRot, period=period, Q0=Q0, dQ=dQ, f=1)
    omega0 = stats.loguniform.rvs(0.1/24., 3./24.)
    sigmaGran = np.random.uniform(20e-6, 200e-6)
    term2 = terms.SHOTerm(sigma=sigmaGran, w0=omega0, Q=2.**0.5)
    gp = celerite2.GaussianProcess(term1 + term2)
    gp.compute(t, yerr=np.zeros(len(t)) + yerr)
    y = gp.sample()
    y += np.random.normal(loc=0., scale=yerr, size=len(y))

    return y, yerr

################################
## Rotation stuff ##

def set_params(params, gp, x, yerr):
    gp.mean = params[0]
    par = params[1:]
    gp.kernel = terms.RotationTerm(sigma=par[0], period=par[1], Q0=par[2], \
                dQ=par[3], f=par[4]) \
                + terms.SHOTerm(sigma=par[5], w0=par[6], Q=2.**0.5)
    gp.compute(x, diag=yerr**2 + par[7], quiet=True)
    return gp

def neg_log_like(params, gp, x, y, yerr):
    gp = set_params(params, gp, x, yerr)
    return -gp.log_likelihood(y)
