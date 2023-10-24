import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from pdb import set_trace

def residual_flare(params, t, y, yerr, complexity=1):
    return (y - exp_line(t, params, complexity=complexity))/yerr

def residual_line(params, t, y, yerr):
    return (y - np.polyval([params['a'], params['b'], params['c']], t))/yerr

def flare_dip_residuals(par, t, data, err, complexity):
    try:
        return (data - (flare_dip(par, t) + exp_convolved(t, par, complexity=complexity) \
                + np.polyval([par['d'], par['e'], par['f']], t)))/err
    except FloatingPointError:
        return np.zeros(len(data)) + 1e6

    #return (data - (dip(par, t) + exp_doubledecay(t, par, complexity=1) \
    #            + np.polyval([par['d'], par['e'], par['f']], t)))/err

def exp_line(t, params, complexity=1):
    return exp_doubledecay(t, params, complexity=complexity) \
         + np.polyval([params['d'], params['e'], params['f']], t)

def gauss_residuals(par, t, data, err):

    gaussleft = np.exp(-(t[t < par['x0']] - par['x0'])**2/(2.*par['w1']**2))
    gaussright = np.exp(-(t[t >= par['x0']] - par['x0'])**2/(2.*par['w2']**2))
    gauss = par['A']*np.concatenate((gaussleft, gaussright))
    return (data - gauss - np.polyval([par['d'], par['e'], par['f']], t))/err

def flare_dip(par, x, plots=False, add_flare=False):

    #val = par.valuesdict()
    x0 = par['x0']

    dipvalid = x >= x0
    #if dip_position == 'pre':
    #    dipvalid = np.logical_and(x >= x0, x < 0.)
    #elif dip_position == 'post':
    #    dipvalid = np.logical_and(x >= x0, x > 0.)
    exponent1 = -(abs(x[dipvalid] - x0)/par['w2'])**par['n']
    exponent1[exponent1 < -100.] = -100.
    exponent2 = -(abs(x[x < x0] - x0)/par['w1'])**par['n']
    exponent2[exponent2 < -100.] = -100.

    y = np.zeros(len(x))
    y[dipvalid] = par['A']*np.exp(exponent1)
    y[x < x0] = par['A']*np.exp(exponent2)

    if add_flare:
        y += exp_line(x, par, complexity=complexity)

    if np.sum(np.isinf(y)) > 0:
        set_trace()

    if plots:
        plt.plot(x, y)
        plt.show()
        set_trace()

    return y

def exp_doubledecay(t, pars, complexity=1, plots=False):
    '''
    Models exponential decay phase with double exponential decay
    (Davenport+2014).

    Parameters
    ----------
    t0: time to start the exponential decay
    t12: time when the impulsive decay turns into gradual decay. This is
        **not** exactly Kowalski's (2013) metric, and is added to t0
        (--> time after the peak)
    A: flux peak
    tau1: decay time scale during impusive phase
    tau2: decay time scale during gradual decay phase
    taurise: impulsive rise time scale

    Returns
    -------
    Sum of exp models
    '''

    mod = np.zeros(len(t))
    for i in np.arange(complexity):
        t0 = pars['t0' + str(i)]
        A = pars['A' + str(i)]
        tau1 = pars['tau1' + str(i)]
        tau2 = pars['tau2' + str(i)]
        taurise = pars['taurise' + str(i)]
        t12 = pars['t12' + str(i)]
        timp = np.logical_and(t >= t0, t <= t12)

        x = abs(t[timp] - t0)/tau1
        x[x > 100.] = 100.
        mod[timp] += A*np.exp(-x)
        x = abs(t12 - t0)/tau1
        if x > 100.:
            x = 100.
        B = A*np.exp(-x)
        x = abs(t[t > t12] - t12)/tau2
        x[x > 100.] = 100.
        mod[t > t12] += B*np.exp(-x)
        x = abs(t[t < t0] - t0)/taurise
        x[x > 100.] = 100.
        mod[t < t0] += A*np.exp(-x)

    if plots:
        plt.close('all')
        plt.plot(t, mod)
        plt.show()
        set_trace()
        plt.close('all')

    return mod

def gaussian_bump(t, par, plots=False):
    '''
    Asymmetric Gaussian function to incldue bumps and flat tops.
    '''

    x0 = par['g_t0']
    dipvalid = t >= x0
    exponent1 = -(abs(t[dipvalid] - x0)/par['g_w2'])**par['g_n']
    exponent1[exponent1 < -100.] = -100.
    exponent2 = -(abs(t[t < x0] - x0)/par['g_w1'])**par['g_n']
    exponent2[exponent2 < -100.] = -100.

    y = np.zeros(len(t))
    y[dipvalid] = par['g_ampl']*np.exp(exponent1)
    y[t < x0] = par['g_ampl']*np.exp(exponent2)

    if plots and par['g_ampl'] > 0.:
        plt.plot(t, y)
        plt.show()
        set_trace()

    return y

def exp_convolved(t, par, complexity=1, plots=False):
    '''
    Convolution of double exponential and Gaussian function, from Mendoza+2022.
    '''

    h = lambda t, B, C, D: np.exp(-D*t + (B/C + D*C/2.)**2) \
                    *special.erfc((B - t)/C + D*C/2.)
    mod = np.zeros(len(t))

    for i in np.arange(complexity):
        # Center time axis on each B_i
        ti = t - par['B' + str(i)]
        bi = 0.

        cc = np.pi**0.5*par['A' + str(i)]*par['C' + str(i)]/2.
        h1 = h(ti, bi, par['C' + str(i)], par['D1' + str(i)])
        h1[np.logical_or(h1 < 1e-6, np.isnan(h1))] = 0.
        h2 = h(ti, bi, par['C' + str(i)], par['D2' + str(i)])
        h2[np.logical_or(h2 < 1e-6, np.isnan(h2))] = 0.
        mod_i = cc*(par['F1' + str(i)]*h1 + (1. - par['F1' + str(i)])*h2)

        if plots:
            plt.close('all')
            plt.plot(t, mod_i)
        mod += mod_i

    if np.sum(np.logical_or(np.isnan(mod), np.isinf(mod))) > 0:
        return np.zeros(len(t)) + 1e6
    #if (mod == 0.).all():
    #    set_trace()

    if plots:
        plt.close('all')
        plt.plot(t, mod, linewidth=3)
        plt.show()
        set_trace()

    return mod

# Mendoza models
def flare_eqn(t, ampl):
    '''
    The equation that defines the shape for the Continuous Flare Model
    '''
    #Values were fit & calculated using MCMC 256 walkers and 30000 steps

    A, B, C, D1, D2, F1 = [0.9688, -0.2513, 0.2268, 0.1555, 1.2151, 0.1270]

    # We include the corresponding errors for each parameter from the MCMC analysis
    A_err, B_err, C_err, D1_err, D2_err, F1_err \
        = [0.0079, 0.0004, 0.0007, 0.0013, 0.0045, 0.0011]

    h = lambda t, B, C, D: np.exp(-D*t + (B/C + D*C/2.)**2) \
                    *special.erfc((B - t)/C + D*C/2.)

    try:
        cc = np.pi**0.5*A*C/2.
        h1 = h(t, B, C, D1)
        h1[np.logical_or(np.isnan(h1), h1 < 1e-10)] = 0.
        h2 = h(t, B, C, D2)
        h2[np.logical_or(np.isnan(h2), h2 < 1e-10)] = 0.
        eqn = cc*(F1*h1 + (1. - F1)*h2)
    except FloatingPointError:
        eqn = np.zeros(len(t))

    return eqn*ampl


def flare_model_mendoza(t, par, complexity=1, plots=False):
    '''
    The Continuous Flare Model evaluated for single-peak (classical) flare events.
    Use this function for fitting classical flares with most curve_fit
    tools. Reference: Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6

    References
    --------------
    Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Jackman et al. (2018) https://arxiv.org/abs/1804.03377

    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over

    tpeak : float
        The center time of the flare peak

    fwhm : float
        The Full Width at Half Maximum, timescale of the flare

    ampl : float
        The amplitude of the flare


    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time

        A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
        and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
    '''

    flare = np.zeros(len(t))

    for c in np.arange(complexity):
        t_new = (t - par['tpeak' + str(c)])/par['fwhm' + str(c)]
        flare += flare_eqn(t_new, par['ampl' + str(c)])

    if plots:
        plt.close('all')
        plt.plot(t, flare)
        plt.show()
        set_trace()

    return flare

def mendoza_residuals(par, t, data, err, complexity=1):
    return (data - (flare_dip(par, t) + flare_model_mendoza(t, par, \
                    complexity=complexity) \
                    + np.polyval([par['d'], par['e'], par['f']], t)))/err
