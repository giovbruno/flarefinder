'''
Class for flares and relative properties.
'''

from lmfit import Parameters, fit_report, minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
from scipy.integrate import trapz
from scipy import signal
from astropy.modeling.physical_models import BlackBody
from astropy import units as u
from astropy import units, constants
from astropy.stats import sigma_clip
from astropy import timeseries
from scipy.stats import ks_2samp
import peakutils.peak as peakut
import copy
import models
from pdb import set_trace

class flare:
    def __init__(self, tdata, ydata, yerrdata, noise_level, \
            traw=None, yraw=None, \
            rednoise=None, tbeg=None, tend=None, \
            tpeak=None, Tstar=5777., Tflare=9000, Rstar=1., wth=None, fth=None):

        self.tdata = tdata*u.day
        self.ydata = ydata
        self.yerrdata = yerrdata
        self.noise_level = noise_level
        self.rednoise = rednoise
        self.traw = traw
        self.yraw = yraw
        self.tbeg = tbeg
        self.tend = tend
        self.tpeak = tpeak
        self.Tstar = Tstar*units.K
        self.Tflare = Tflare*units.K
        self.Rstar = Rstar*constants.R_sun
        self.wth = wth
        self.fth = fth

    def get_model_from_result(self, fit_result):
        return self.ydata - fit_result.residual*self.yerrdata

    def fit_line(self):
        '''
        Polynomial fit to the flare profle.
        '''

        fit_params_line = Parameters()
        fit_params_line.add('a', value=0., vary=True)
        fit_params_line.add('b', value=0., vary=True)
        fit_params_line.add('c', value=0., vary=True)

        tt = self.tdata.value
        result = minimize(models.residual_line, fit_params_line, \
            calc_covar=False, args=(tt, self.ydata, self.yerrdata), \
            method='least_squares', nan_policy='omit')
        print('0 flares: AIC = {:.2f}, redchi2 = {:.2f}'.format( \
                result.aic, result.redchi))

        self.result_line = result

        return

    def fit_gaussian(self, npoly, plots=False):
        '''
        Gaussian fit to the flare profile (to be compared with single
        exponential).

        Parameters
        ----------
        npoly: degree for the polynomial fit to the continuum level.
        '''

        tt = self.tdata.value

        fit_params = Parameters()
        fit_params.add('d', value=0., vary=False)
        fit_params.add('e', value=0., vary=False)
        fit_params.add('f', value=0., vary=False)
        if npoly == 0:
            fit_params['f'].vary = True
        if npoly == 1:
            fit_params['e'].vary = True
            fit_params['f'].vary = True
        if npoly == 2:
            fit_params['d'].vary = True
            fit_params['e'].vary = True
            fit_params['f'].vary = True

        fit_params.add('A', value=self.ydata.max(), vary=True)
        fit_params.add('w1', value=1e-3, vary=True)
        fit_params.add('w2', expr='w1')
        fit_params.add('x0', value=0., vary=True)
        #fit_params.add('n', value=2., min=2., max=4., vary=False)

        fit_kws = {}
        fit_kws['niter'] = 10
        fit_kws['minimizer_kwargs'] = {}
        fit_kws['minimizer_kwargs']['method'] = 'powell'
        try:
            result = minimize(models.gauss_residuals, fit_params, \
            calc_covar=False, args=(tt, self.ydata, self.yerrdata), \
            method='basinhopping', nan_policy='omit')#, **fit_kws)
        except TypeError:
            set_trace()
        print('Gaussian fit: AIC = {:.2f}, redchi2 = {:.2f}'.format( \
                result.aic, result.redchi))

        self.result_gauss = result

        if plots:
            plt.close('all')
            plt.plot(tt, self.ydata)
            plt.plot(tt, self.ydata - result.residual*self.yerrdata)
            plt.show()
            set_trace()

        return

    def fit_flare_profile_davenport(self, complexity, fit_continuum=True, plots=False):
        '''
        Fit multiple flare profiles to the data.
        '''

        if not 'result_line' in dir(self):
            print('You must run fit_line to run the AIC tests.')
            return

        noiselev = self.yerrdata.max()
        tt = self.tdata.value

        bestaic = self.result_line.aic
        self.npeaks = 0
        for n in np.arange(1, complexity + 1):
            fit_params = Parameters()
            for nf in np.arange(n):
                fit_params.add('t0' + str(nf), value=1e-2*nf, \
                        min=self.tbeg, max=self.tend)
                fit_params.add('t12' + str(nf) + '_diff', \
                        value=1e-2*(nf + 1), min=0., max=tt.max()*0.7)
                # t12 must be > t0, so an additional parameter is fitted instead
                fit_params.add('t12' + str(nf), expr='t0' + str(nf) \
                            + ' + t12' + str(nf) + '_diff')
                fit_params.add('A' + str(nf), value=noiselev)#, min=3.*noiselev)

                # Try to keep main flare duration even when adding new ones
                if n > 1 and nf != 0:
                    fit_params.add('tau1' + str(nf), value=4e-3, \
                                min=np.diff(tt).min()*3., max=0.1)
                elif n > 1 and nf == 0:
                    fit_params.add('tau1' + str(nf),    \
                        value=result.params['tau10']*1., \
                        min=result.params['tau10']*0.2, max=0.1)
                else:
                    fit_params.add('tau1' + str(nf), value=4e-3, \
                                    min=np.diff(tt).min()*3., max=0.1)
                # Require tau1 < tau2
                fit_params.add('taumult' + str(nf), value=1.1, min=1., max=100.)
                fit_params.add('tau2' + str(nf),  \
                        expr='tau1' + str(nf) + ' * taumult' + str(nf), \
                        max=np.ptp(tt)/3.)
                fit_params.add('taurise_mult' + str(nf), value=0.1, \
                            min=1e-6, max=0.5)
                fit_params.add('taurise' + str(nf), expr='tau1' + str(nf) \
                            + ' * taurise_mult' + str(nf))
            fit_params.add('d', value=0., vary=False)#, min=-0.1, max=0.1, vary=True)
            fit_params.add('e', value=0., vary=False)#, min=-0.1, max=0.1, vary=True)
            #fit_params.add('f', value=0., min=-0.001, max=0.001, vary=True)
            fit_params.add('f', value=0., vary=True)#, \
                #min=np.median(self.ydata) - 3.*np.median(self.yerrdata), \
                #max=np.median(self.ydata) + 3.*np.median(self.yerrdata))
            if not fit_continuum:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = False
            # Add Gaussian (also flattened) for both peak-bump and flat-top profiles
            fit_params.add('g_ampl', value=4.*noiselev)#, min=3*noiselev, max=1.)
            #fit_params.add('g_t0', value=1e-3, min=-1e-3, max=tw.max()*0.7)
            fit_params.add('g_t0', value=1e-3, min=self.tbeg, max=self.tend/2.)
            fit_params.add('g_w1', min=1e-3, max=(self.tend - self.tbeg)/5.)
            fit_params.add('g_w2', min=1e-3, max=(self.tend - self.tbeg)/5.)
            fit_params.add('g_n', value=2.1, min=2.0, max=8.)
            # Parameters for dip
            fit_params.add('A', value=0., min=-1, max=0., vary=False)
            if tt.min() != self.tbeg:
                fit_params.add('x0', value=-1e-3, min=tt.min(), \
                        max=self.tbeg, vary=False)
            else:
                fit_params.add('x0', value=-1e-3, min=tt.min(), \
                        max=tt.min() + np.diff(tt).min(), vary=False)
            fit_params.add('w1', value=3./(24.*60), min=0.1/(24.*60), \
                                max=20./(24.*60), vary=False)
            fit_params.add('w2', value=3./(24.*60), min=0.1/(24.*60), \
                                max=20./(24.*60), vary=False)
            fit_params.add('n', value=2.1, min=2., max=4., vary=False)

            result = minimize(models.flare_dip_residuals, fit_params, \
                    args=(tt, self.ydata, self.yerrdata), \
                    kws={'complexity': n}, method='powell', \
                    nan_policy='omit', calc_covar=False)
            print(n, 'flares: AIC = {:.2f}, redchi2 = {:.2f}'.format( \
                    result.aic, result.redchi))

            # If the fit does not improve while adding flares, stop here
            if result.aic >= self.result_line.aic - 6:
                self.result_nodip = result
                self.bump_delta_aic = -np.inf
                if plots:
                    model = models.exp_line(tt, result.params, complexity=1)
                    plt.plot(tt, self.ydata)
                    plt.plot(tt, model)
                    plt.show()
                    set_trace()

                return

            else:
                if result.aic < bestaic - 6.:
                    bestaic = np.copy(result.aic)
                    self.npeaks += 1
                    X = copy.deepcopy(result)
                elif result.aic > bestaic - 6. and n > 1:
                    break

            if n == complexity:
                break

        # Is the Gaussian bump needed?
        #X2 = copy.deepcopy(X)
        #X2.params['g_ampl'].min = 0.
        #X2.params['g_ampl'].value = 0.
        #for p in ['g_t0', 'g_ampl', 'g_w1', 'g_w2', 'g_n']:
        #    X2.params[p].vary = False
        #result_nobump = minimize(models.dip_residuals, X2.params, \
        #    args=(tt, self.ydata, self.yerrdata), \
        #    kws={'complexity': self.npeaks}, method='powell', \
        #    nan_policy='omit', calc_covar=False)
        #self.bump_delta_aic =   result_nobump.aic - X.aic
        #if self.bump_delta_aic < 6:
        #    X = copy.deepcopy(result_nobump)

        ## Are exponential flares needed?
        #X3 = copy.deepcopy(X)
        #for p in ['g_t0', 'g_ampl', 'g_w1', 'g_w2', 'g_n']:
        #    X3.params[p].vary = True
        #for p in X3.params.keys():
        #    if str(p) not in ['g_t0', 'g_ampl', 'g_w1', 'g_w2', 'g_n']:
        #        if 'A' in p:
        #            X3.params[p].min = -10.
        #            X3.params[p].value = 0.
        #        X3.params[p].vary = False
        #result_noexp = minimize(models.dip_residuals, X3.params, \
        #    args=(tt, self.ydata, self.yerrdata), \
        #    kws={'complexity': self.npeaks}, method='powell', \
        #    nan_policy='omit', calc_covar=False)
        #self.noexp_delta_aic = result_noexp.aic - X.aic
        #if self.noexp_delta_aic < 6:
        #    X = copy.deepcopy(result_noexp)

        self.result_nodip = X

        if plots:
            model = models.exp_line(tt, X.params, complexity=self.npeaks)
            plt.plot(tt, self.ydata)
            plt.plot(tt, model)
            plt.show()
            set_trace()
            plt.close()

        return

    def fit_flare_profile(self, complexity, threshold, \
            fit_continuum=False, get_uncertaintie_mcmc=False, plots=False, \
            min_datapoints=3, min_dist=5, plotname='', plot_scaled_time=True):
        '''
        Fit multiple flare profiles to the data, using Mendoza's model.
        Use Mendoza's tempate
        '''

        #if get_uncertainties:
        #    fit_method = 'powell'
        #else:
        #    fit_method = 'least_squares'

        if not 'result_line' in dir(self):
            print('You must first run fit_line to run the AIC tests.')
            return
        #if not 'result_gauss' in dir(self):
        #    print('You must first run fit_gaussian to run the AIC tests.')
        #    return

        # Rescale time to t1/2
        fmax = self.ydata.argmax()
        tmax = self.tdata[fmax]
        diffs = abs(self.ydata - 0.5*self.ydata.max())
        checkleft = self.tdata < tmax
        checkright = self.tdata >= tmax
        if np.sum(checkleft) == 0 or np.sum(checkright) == 0:
            print('Too close to a data set border')
            self.npeaks = 0.
            return
        x1 = self.tdata[diffs[checkleft].argmin()]
        bb = np.sum(self.tdata < tmax)
        x2 = self.tdata[diffs[checkright].argmin() + bb]
        t12 = x2 - x1
        self.t12 = t12
        tt = self.tdata.value

        noiselev = self.noise_level

        bestaic = self.result_line.aic
        self.npeaks = 0
        daytomin = 24*60.
        for n in np.arange(1, complexity + 1):
            fit_params = Parameters()
            for nf in np.arange(n):
                fit_params.add('ampl' + str(nf), min=self.noise_level*3, \
                        max=5*self.ydata.max())#, value=self.ydata.max())
                fit_params.add('tpeak' + str(nf), \
                        min=self.tbeg, max=self.tend)#, value=0.)
                '''
                Reduced minimum FWHM from twice to 1/10th data cadence, just
                to set a lower limit.
                '''
                fit_params.add('fwhm' + str(nf) , \
                        min=np.diff(tt).min()*0.1, max=self.tend - self.tbeg)
                        #value=self.t12.value)
                        #max=5.*self.t12.value, \

            fit_params.add('d', value=0., vary=True)#, min=-0.1, max=0.1)
            fit_params.add('e', value=0., vary=True)#, min=-0.1, max=0.1)
            fit_params.add('f', value=0., vary=True, min=-noiselev, \
                                                    max=noiselev)

            if fit_continuum == -1:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = False
            elif fit_continuum == 0:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = True
            elif fit_continuum == 1:
                fit_params['d'].vary = False
                fit_params['e'].vary = True
                fit_params['f'].vary = True
            elif fit_continuum == 2:
                fit_params['d'].vary = True
                fit_params['e'].vary = True
                fit_params['f'].vary = True

            # Parameters for dip
            fit_params.add('A', value=0., min=-1, max=0., vary=False)
            #if tt.min() != (self.tbeg/self.t12).value:
            if tt.min() != self.tbeg:
                fit_params.add('x0', min=tt.min(), \
                        max=self.tbeg, vary=False)
            else:
                fit_params.add('x0', min=tt.min(), \
                    max=tt.min() + np.diff(tt).min(), vary=False)

            fit_params.add('w1', value=2./daytomin, min=0.1/daytomin, \
                        max=20./daytomin, vary=False)
            fit_params.add('w2', value=2./daytomin, min=0.1/daytomin, \
                        max=20./daytomin, vary=False)
            fit_params.add('n', value=2.1, min=2., max=4., vary=False)
            '''
            # Repeat minimization to try and increase the explored parameter
            # space
            for iter in np.arange(10):
                scramble = True
                scramble_i = 0
                while scramble:
                    if scramble_i > 0:
                        for p in fit_params:
                            if p != 'd' and p != 'e' and p != 'f':
                                fit_params[p].value = np.random.uniform( \
                                    low=fit_params[p].min + 1e-5, \
                                    high=fit_params[p].max - 1e-5)
                    try:
                        result_ = minimize(models.mendoza_residuals, fit_params, \
                           args=(tt, self.ydata, self.yerrdata), \
                           kws={'complexity': n}, method='powell', \
                           nan_policy='omit', calc_covar=True)
                        scramble = False
                        break
                    except FloatingPointError:
                        scramble_i += 1
                        pass

                if iter == 0 or result_.redchi < chi2min:
                    chi2min = result_.redchi
                    result = copy.deepcopy(result_)
                #elif iter > 0 and :
                #    chi2min = result_.redchi
                #    result = copy.deepcopy(result_)
            '''
            fit_kws = {}
            fit_kws['niter'] = 10
            fit_kws['minimizer_kwargs'] = {}
            fit_kws['minimizer_kwargs']['method'] = 'powell'
            # If the minimization fails, change the initial parameters
            scramble = True
            scramble_i = 0
            while scramble:
                try:
                    result = minimize(models.mendoza_residuals, fit_params, \
                       args=(tt, self.ydata, self.yerrdata), \
                       kws={'complexity': n}, method='basinhopping', \
                       nan_policy='omit', calc_covar=True, **fit_kws)
                    scramble = False
                except FloatingPointError:
                    for p in fit_params:
                        if p != 'd' and p != 'e' and p != 'f':
                            fit_params[p].value = np.random.uniform( \
                                low=fit_params[p].min + 1e-5, \
                                high=fit_params[p].max - 1e-5)
                    scramble_i += 1
                    if scramble_i > 100:
                        self.npeaks = 0
                        break
                    else:
                        pass

            print(n, 'flares: AIC = {:.2f}, redchi2 = {:.2f}'.format( \
                    result.aic, result.redchi))

            # If the fit fails after the first flare, stop here
            if n > 1 and result.success == False:
                break
            if n == 1 and result.aic >= self.result_line.aic - 6:
                self.result_nodip = result
                break
            elif n > 1 and result.aic > bestaic - 6.:
                break
            else:
                if n > 1:
                    # If the additional flare model does not have a SNR > 4, stop here
                    thisflare = self.ydata - result.residual*self.yerrdata
                    if thisflare.max() < threshold*noiselev:
                        break
                    # If the flare peak is < min_dist points close to another
                    # peak, stop here
                    t_peaks_sol = []
                    tooclose = False
                    for tp in np.arange(self.npeaks + 1):
                        t_peaks_sol.append(result.params['tpeak' + str(tp)])
                        if (np.diff(np.sort(t_peaks_sol)) \
                                        < np.diff(tt).min()*min_dist).any():
                            tooclose = True
                    if tooclose:
                        break
                    # if the flare model lasts less than min_datapoints, stop here
                    #try:
                    flare_start = tt[thisflare > noiselev][0]
                    flare_end = tt[thisflare > noiselev][-1]
                    if flare_end - flare_start < min_datapoints*np.diff(tt).min():
                        break
                    #except IndexError:
                    #    break
                bestaic = np.copy(result.aic)
                self.npeaks += 1
                self.result_nodip = copy.deepcopy(result)
                #lastmodel = self.get_single_profile(n - 1)/self.ydata.max()
                # Additional peaks? If yes, continue
                #newpeaks = list(peakut.indexes(result.residual*self.yerrdata, \
                #        thres=threshold*np.std(result.residual*self.yerrdata), \
                #        min_dist=10, thres_abs=True))
                #if len(newpeaks) == 0:
                #    print('No more peaks')
                #    break

            if n == complexity:
                break

        # Use an MCMC to estimate uncertainties
        #if get_uncertainties_mcmc:
        #    count_scramble = 0
        #    scramble = True
        #    # Varying the initial params might help
        #    result2 = copy.deepcopy(result)
        #    while scramble:
        #        try:
        #            result = minimize(models.mendoza_residuals, result2.params, \
        #                 args=(tt, self.ydata, self.yerrdata), \
        #                 kws={'complexity': n}, method='emcee', \
        #                 nan_policy='omit', is_weighted=True, progress=True)
        #            scramble = False
        #        except FloatingPointError:
        #            count_scramble += 1
        #            print('Scramble #', count_scramble)
        #            for nfl in np.arange(self.npeaks + 1):
        #                for pfl in ['ampl', 'fwhm']:
        #                    result2.params[pfl + str(nfl)].value \
        #                        += np.random.uniform(low=0, \
        #                        high=1e-3*result2.params[pfl + str(nfl)].value)
        #                    result2.params[pfl + str(nf)].min \
        #                        = 0.9*result2.params[pfl + str(nf)].value
        #                    result2.params[pfl + str(nf)].max \
        #                        = 1.1*result2.params[pfl + str(nf)].value
        #        if count_scramble > 100:
        #            # If the fit fails, stop here, and append a message
        #           self.npeaks = 0
        #           return
        #    self.result_nodip = copy.deepcopy(result)
        X = copy.deepcopy(self.result_nodip)

        # Save flare components properties
        if self.npeaks > 0:
            self.durations = []
            for nf in np.arange(self.npeaks):
                try:
                    self.durations.append(self.get_single_duration(nf))
                except IndexError:
                    self.durations.append(0.)

        # Periodogram of fit residuals (not normalized)
        ls = timeseries.LombScargle(self.tdata*u.day.to(u.s), X.residual \
                        *self.yerrdata*self.ydata.max(), \
                        normalization='standard', fit_mean=True)
        self.ls_residuals = ls

        if plots:
            plt.close('all')
            print(X.params.pretty_print())
            #model = models.exp_convolved(tt, X.params, complexity=self.npeaks, \
            #            plots=False)
            model = models.flare_model_mendoza(tt, X.params, \
                    complexity=self.npeaks, plots=False)
            fig, axs = plt.subplots(nrows=2, sharex=True)
            axs[0].plot(tt, self.ydata, 'k', alpha=1.)
            model = self.ydata - X.residual*self.yerrdata
            #plt.errorbar(tt, self.ydata, yerr=self.yerrdata, fmt='k', \
            #            alpha=0.5, errorevery=10)
            axs[0].plot(tt, model, 'r', linewidth=3)
            axs[0].set_ylabel('Relative flux', fontsize=14)
            axs[1].plot(tt, self.result_nodip.residual*self.yerrdata, 'k')
            axs[1].set_xlabel(r'Relative time [$t_{1/2}$]', fontsize=14)
            axs[1].set_ylabel(r'Residuals', fontsize=14)
            plt.tight_layout()
            if plotname == '':
                plt.show()
                set_trace()
            else:
                plt.savefig(plotname)
            plt.close()

        return

    def fit_QPP(self, plots=False, verbose=False, plotname=''):
        '''
        Subtract a one-flare profile from a complex flare, get maximum
        oscillation amplitude.
        '''

        self.fit_flare_profile(1, 4., plots=plots, plotname=plotname)
        oscill_ampl = 0.5*np.ptp(self.result_nodip.residual*self.yerrdata)
        oscill_ampl_ratio = oscill_ampl/self.result_nodip.params['ampl0']
        if verbose:
            print('Oscillation amplitude:', oscill_ampl)

        return oscill_ampl_ratio

    def fit_flare_profile_mendoza_noconstraints(self, complexity, threshold, \
            fit_continuum=False, get_uncertainties=False, plots=False):
        '''
        Fit multiple flare profiles to the data, using Mendoza's model - which
        should avoid the need for a Gaussian.
        '''

        if not 'result_line' in dir(self):
            print('You must first run fit_line to run the AIC tests.')
            return
        if not 'result_gauss' in dir(self):
            print('You must first run fit_gaussian to run the AIC tests.')
            return

        # Rescale time to t1/2
        fmax = self.ydata.argmax()
        tmax = self.tdata[fmax]
        diffs = abs(self.ydata - 0.5*self.ydata.max())
        checkleft = self.tdata < tmax
        checkright = self.tdata >= tmax
        if np.sum(checkleft) == 0 or np.sum(checkright) == 0:
            print('Too close to a data set border')
            self.npeaks = 0.
            return
        x1 = self.tdata[diffs[checkleft].argmin()]
        bb = np.sum(self.tdata < tmax)
        x2 = self.tdata[diffs[checkright].argmin() + bb]
        t12 = x2 - x1
        self.t12 = t12

        tt = (self.tdata/t12).value

        y = self.ydata/self.ydata.max()
        #noiselev = self.yerrdata.max()/self.ydata.max()
        noiselev = self.noise_level/self.ydata.max()
        yerr = self.yerrdata/self.ydata.max()

        bestaic = self.result_line.aic
        self.npeaks = 0

        for n in np.arange(1, complexity + 1):
            fit_params = Parameters()
            for nf in np.arange(n):
                #if nf == 0:
                fit_params.add('A' + str(nf), min=1e-6, max=100., value=1., vary=True)
                    #fit_params.add('A' + str(nf), min=hreshold*noiselev, \
                    #             max=100., value=1., vary=True)
                #else:
                #    fit_params.add('A' + str(nf), min=threshold*np.std(result.residual*yerr), max=100., vary=True)
                if nf == 0:
                    fit_params.add('B' + str(nf), value=0., \
                            min=max([-10, tt.min()]), max=tt.max())
                            #min=tt.min(), max=tt.max())
                else:
                    fit_params.add('B' + str(nf), \
                            min=result.params['B' + str(nf - 1)] \
                            + np.diff(tt).min()*10., max=tt.max())
                #if nf == 0:
                cmin = 0.001/(24.*60.)/self.t12.value
                cmax = 10./(24*60.)/self.t12.value
                fit_params.add('C' + str(nf), min=0., value=0.1, max=1.)
                #else:
                #    fit_params.add('C' + str(nf), min=0.1, max=10.)
                d1min = 0.001/(24.*60.)/self.t12.value
                d1max = 10./(24.*60.)/self.t12.value
                fit_params.add('D1' + str(nf), min=0., max=1., \
                        value=0.1)
                #d1addmax = tt.max()/5.
                fit_params.add('D1add' + str(nf), min=0., max=3., \
                            value=1.)
                #fit_params.add('D2' + str(nf), min=1e-5, max=self.t12.value, value=0.1)
                fit_params.add('D2' + str(nf), expr='D1' + str(nf) \
                            + ' + D1add' + str(nf))
                fit_params.add('F1' + str(nf), min=0., max=1., value=0.3)

            fit_params.add('d', value=0., vary=True)#, min=-0.1, max=0.1, vary=True)
            fit_params.add('e', value=0., vary=True)#min=-0.1, max=0.1,
            fit_params.add('f', value=0., vary=True, \
                min=-noiselev, max=noiselev)

            if fit_continuum == -1:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = False
            elif fit_continuum == 0:
                fit_params['d'].vary = False
                fit_params['e'].vary = False
                fit_params['f'].vary = True
            elif fit_continuum == 1:
                fit_params['d'].vary = False
                fit_params['e'].vary = True
                fit_params['f'].vary = True
            elif fit_continuum == 2:
                fit_params['d'].vary = True
                fit_params['e'].vary = True
                fit_params['f'].vary = True

            # Parameters for dip
            fit_params.add('A', value=0., min=-1, max=0., vary=False)
            if tt.min() != (self.tbeg/self.t12).value:
                fit_params.add('x0', value=-1, min=tt.min(), \
                        max=(self.tbeg/self.t12).value, vary=False)
            else:
                fit_params.add('x0', value=-1, min=tt.min(), \
                    max=(tt.min() + np.diff(tt).min())/self.t12.value, vary=False)
            fit_params.add('w1', value=2., min=1., max=10., vary=False)
            fit_params.add('w2', value=2., min=1., max=10., vary=False)
            fit_params.add('n', value=2.1, min=2., max=4., vary=False)

            fit_kws = {}
            fit_kws['niter'] = 3
            fit_kws['disp'] = False
            fit_kws['minimizer_kwargs'] = {}
            fit_kws['minimizer_kwargs']['method'] = 'powell'
            try:
                result = minimize(models.flare_dip_residuals, fit_params, \
                   args=(tt, y, yerr), kws={'complexity': n}, method='powell', \
                   nan_policy='omit')#, **fit_kws)
            except ValueError:
                if n == 1:
                    result = copy.deepcopy(self.result_line)
                    result.aic = np.inf
                else:
                    break
            except FloatingPointError:
                if n == 1:
                    result = copy.deepcopy(self.result_line)
                    result.aic = np.inf
                else:
                    break
            print(n, 'flares: AIC = {:.2f}, redchi2 = {:.2f}'.format( \
                    result.aic, result.redchi))

            # If the fit does not improve while adding flares, stop here
            if n == 1 and result.aic >= self.result_line.aic - 6 \
                        or result.aic >= self.result_gauss.aic - 6:
                self.result_nodip = result
                break
            elif n > 1 and result.aic > bestaic - 6.:
                break
            else:
                if n > 1:
                    # If the additional flare model does not have a SNR > 4, stop here
                    thisflare = y - result.residual*yerr
                    if thisflare.max() < threshold*noiselev:
                        break
                    # if the flare model lasts less than 5 data points, stop here
                    try:
                        flare_start = tt[thisflare > noiselev][0]
                        flare_end = tt[thisflare > noiselev][-1]
                        if flare_end - flare_start < 5.*np.diff(tt).min():
                            break
                    except IndexError:
                        break
                bestaic = np.copy(result.aic)
                self.npeaks += 1
                self.result_nodip = copy.deepcopy(result)
                #lastmodel = self.get_single_profile(n - 1)/self.ydata.max()
                # Additional peaks? If yes, continue
                newpeaks = list(peakut.indexes(result.residual*yerr, \
                        thres=threshold*np.std(result.residual*yerr), \
                        min_dist=10, thres_abs=True))
                if len(newpeaks) == 0:
                    print('No more peaks')
                    break

            #else:
            #    if result.aic < bestaic - 6. and n == 1:
            #        bestaic = np.copy(result.aic)
            #        self.npeaks += 1
            #        self.result_nodip = copy.deepcopy(result)
            #    elif (result.aic > bestaic - 6. and n > 1:
            #        break

            if n == complexity:
                break

        # Use an MCMC to estimate uncertainties
        if get_uncertainties:
            result = minimize(models.flare_dip_residuals, self.result_nodip.params, \
               args=(tt, y, yerr), kws={'complexity': n}, method='emcee', \
               nan_policy='omit', is_weighted=True)
            self.result_nodip = copy.deepcopy(result)
        X = copy.deepcopy(self.result_nodip)

        # Periodogram of fit residuals (not normalized)
        ls = timeseries.LombScargle(self.tdata*u.day.to(u.s), X.residual*yerr*self.ydata.max(), \
                        normalization='standard', fit_mean=True)
        self.ls_residuals = ls

        if plots:
            plt.close('all')
            print(X.params.pretty_print())
            model = models.exp_convolved(tt, X.params, complexity=self.npeaks, \
                        plots=False)
            plt.plot(tt, y, 'k', alpha=0.4)
            model = y - X.residual*yerr
            #plt.errorbar(tt, self.ydata, yerr=self.yerrdata, fmt='k', \
            #            alpha=0.5, errorevery=10)
            plt.plot(tt, model, linewidth=3)
            plt.xlabel(r'Relative time [$t_{1/2}$]', fontsize=14)
            plt.ylabel('Relative flux', fontsize=14)
            plt.show()
            set_trace()
            plt.close()

        return

    def dip_fit_mendozafree(self, xmin=0., xmax=0., get_uncertainties=False, \
                plots=False):
        '''
        Fix all parameters except the continuum, compare models on this.
        '''
        if not 'result_nodip' in dir(self):
            print('You must first run fit_flare_profile to run the AIC tests.')
            return

        tt = (self.tdata/self.t12).value
        y = self.ydata/self.ydata.max()
        yerr = self.yerrdata/self.ydata.max()

        params = copy.deepcopy(self.result_nodip.params)
        # Compare model with and without dip
        for p in params.keys():
            if p != 'f' and p != 'e' and p != 'd':
                params[p].vary = False
        params['f'].vary = True
        #params['f'].min = -1e-6#np.std(y)
        #params['f'].max = 1e-6#np.std(y)

        result_nodip = minimize(models.flare_dip_residuals, params, \
            args=(tt, y, yerr), kws={'complexity': self.npeaks}, \
            method='powell', nan_policy='omit', calc_covar=False)

        for p in result_nodip.params.keys():
            if str(p) not in ['A', 'x0', 'w1', 'w2', 'n']:
                params[p].vary = False
            else:
                params[p].vary = True

        #params['f'].value = np.random.normal(loc=result_nodip.params['f'].value, \
        #    scale=result_nodip.params['f'].stderr)

        # Attempt several initial pars for the dip position
        chi2min = np.inf
        params['x0'].vary = True
        for iter in np.arange(10):
            if xmin == 0. and xmax == 0.:
                params['x0'].value = np.random.uniform( \
                low=params['x0'].min, high=params['x0'].max)
            else:
                params['x0'].min = xmin
                params['x0'].max = xmax
                params['x0'].value = np.random.uniform(low=xmin, high=xmax)

            fit_kws = {}
            fit_kws['niter'] = 10
            fit_kws['minimizer_kwargs'] = {}
            fit_kws['minimizer_kwargs']['method'] = 'powell'
            result_dip_ = minimize(models.flare_dip_residuals, params, \
                args=(tt, y, yerr), kws={'complexity': self.npeaks}, \
                method='least_squares', nan_policy='omit')#, calc_covar=False, **fit_kws)
            # Update solution if chi2 improves
            if iter == 0 or result_dip.redchi < chi2min:
                result_dip = copy.deepcopy(result_dip_)

        set_trace()
        dip_delta_aic = result_nodip.aic - result_dip.aic

        if get_uncertainties and dip_delta_aic > 6.:
            try:
                result_dip = minimize(models.flare_dip_residuals, \
                    result_dip.params, args=(tt, y, yerr), \
                    kws={'complexity': self.npeaks}, \
                    method='emcee', nan_policy='omit', is_weighted=True)
            except FloatingPointError:
                self.dip_uncertainties = 'LM'

        # Compute significance on the basis of correlated noise level (see Pont+2006)
        dip_width = (result_dip.params['w1'] + result_dip.params['w2']) \
                    /abs(np.diff(tt)[np.diff(tt) != 0.]).min()
        rednoise_level = self.rednoise[1][ \
            abs(np.array(self.rednoise[0]) - dip_width).argmin()]
        significance = abs(result_dip.params['A'].value*self.ydata.max()) \
                    /rednoise_level

        if plots:
            plt.figure()
            plt.plot(tt, y)
            #model = models.asymmetric_gauss(result_dip.params, \
            #        self.tdata.value, self.npeaks)
            model = y - result_dip.residual*yerr
            plt.plot(tt, model)
            plt.xlabel(r'Relative time [$t_{1/2}$]', fontsize=14)
            plt.ylabel('Relative flux', fontsize=14)
            plt.show()
            set_trace()
            plt.close()

        return result_dip, dip_delta_aic, significance

    def dip_fit(self, xmin=0., xmax=0., get_uncertainties_mcmc=False, \
                plots=False):
        '''
        Fix all parameters except the continuum, compare models on this.
        '''
        if not 'result_nodip' in dir(self):
            print('You must first run fit_flare_profile to run the AIC tests.')
            return

        tt = self.tdata.value

        params = copy.deepcopy(self.result_nodip.params)
        # Compare model with and without dip
        for p in params.keys():
            if p != 'f' and p != 'e' and p != 'd':
                params[p].vary = False
        params['f'].vary = True
        #params['f'].min = -1e-6#np.std(y)
        #params['f'].max = 1e-6#np.std(y)

        fit_kws = {}
        fit_kws['niter'] = 10
        fit_kws['minimizer_kwargs'] = {}
        fit_kws['minimizer_kwargs']['method'] = 'powell'
        result_nodip = minimize(models.mendoza_residuals, params, \
           args=(tt, self.ydata, self.yerrdata), \
           kws={'complexity': self.npeaks}, method='basinhopping', nan_policy='omit', \
           calc_covar=True, **fit_kws)

        for p in result_nodip.params.keys():
            if str(p) not in ['A', 'x0', 'w1', 'w2', 'n']:
                params[p].vary = False
            else:
                params[p].vary = True

        # Attempt several initial pars for the dip position
        chi2min = np.inf
        params['x0'].vary = True
        scramble = True
        scramble_i = 0
        while scramble:
            if xmin == 0. and xmax == 0.:
                params['x0'].value = np.random.uniform( \
                low=params['x0'].min, high=params['x0'].max)
            else:
                params['x0'].min = xmin
                params['x0'].max = xmax
                params['x0'].value = np.random.uniform(low=xmin, high=xmax)

            fit_kws = {}
            fit_kws['niter'] = 10
            fit_kws['minimizer_kwargs'] = {}
            fit_kws['minimizer_kwargs']['method'] = 'powell'
            try:
                result_dip_ = minimize(models.mendoza_residuals, params, \
                args=(tt, self.ydata, self.yerrdata), \
                kws={'complexity': self.npeaks}, \
                method='basinhopping', nan_policy='omit', calc_covar=True, **fit_kws)
                #method='least_squares', nan_policy='omit', calc_covar=True)
                scramble = False
            except ValueError:
                set_trace()
            except FloatingPointError:
                for p in params:
                    params[p].value = np.random.uniform( \
                        low=params[p].min + 1e-5, high=params[p].max - 1e-5)
                scramble_i += 1
                if scramble_i > 100:
                    result_dip_ = copy.deepcopy(result_nodip)
                    break
                else:
                    pass

            # Update solution if chi2 improves
            if iter == 0 or result_dip_.redchi < chi2min or scramble == False:
                result_dip = copy.deepcopy(result_dip_)

        dip_delta_aic = result_nodip.aic - result_dip.aic
        if get_uncertainties_mcmc and dip_delta_aic > 6.:
            try:
                result_dip = minimize(models.flare_dip_residuals, \
                    result_dip.params, args=(tt, y, yerr), \
                    kws={'complexity': self.npeaks}, \
                    method='emcee', nan_policy='omit', is_weighted=True)
            except FloatingPointError:
                self.dip_uncertainties = 'LM'

        # Compute significance on the basis of correlated noise level (see Pont+2006)
        dip_width = (result_dip.params['w1'] + result_dip.params['w2']) \
                    /abs(np.diff(tt)[np.diff(tt) != 0.]).min()
        rednoise_level = self.rednoise[1][ \
            abs(np.array(self.rednoise[0]) - dip_width).argmin()]
        significance = abs(result_dip.params['A'].value/rednoise_level)

        if plots:
            plt.figure()
            plt.plot(tt, self.ydata)
            #model = models.asymmetric_gauss(result_dip.params, \
            #        self.tdata.value, self.npeaks)
            model = self.ydata - result_dip.residual*self.yerrdata
            plt.plot(tt, model)
            plt.xlabel(r'Relative time [$t_{1/2}$]', fontsize=14)
            plt.ylabel('Relative flux', fontsize=14)
            plt.show()
            set_trace()
            plt.close()

        return result_dip, dip_delta_aic, significance

    def get_single_profile(self, nsol, model='mendoza_constrained', \
            double_t=True):
        '''
        Parameters
        ----------
        double_t: whether to double the length of the time axis in order to
        avoid cut flare profiles.
        '''

        if 'result_nodip' not in dir(self):
            print('You must run a fit first.')
            return

        if double_t:
            tt = np.arange(self.tdata.min().value, self.tdata.max().value*2, \
                    np.diff(self.tdata).min().value)
        else:
            tt = np.copy(self.tdata.value)

        param_i = Parameters()
        X = copy.deepcopy(self.result_nodip)

        if model == 'davenport':
            param_i.add('t00', value=X.params['t0' + str(nsol)].value)
            param_i.add('A0', value=X.params['A' + str(nsol)].value)
            param_i.add('tau10', value=X.params['tau1' + str(nsol)].value)
            param_i.add('tau20', value=X.params['tau2' + str(nsol)].value)
            param_i.add('t120', value=X.params['t12' + str(nsol)].value)
            param_i.add('taurise0', value=X.params['taurise' + str(nsol)].value)
            thisflare = models.exp_doubledecay(self.tdata.value, param_i, \
                    complexity=1)

        elif model == 'mendoza_free':
            tt = (self.tdata/self.t12).value
            param_i.add('A0', value=X.params['A' + str(nsol)].value)
            param_i.add('B0', value=X.params['B' + str(nsol)].value)
            param_i.add('C0', value=X.params['C' + str(nsol)].value)
            param_i.add('D10', value=X.params['D1' + str(nsol)].value)
            param_i.add('D20', value=X.params['D2' + str(nsol)].value)
            param_i.add('F10', value=X.params['F1' + str(nsol)].value)
            try:
                thisflare = models.exp_convolved(tt, param_i, complexity=1)
            except FloatingPointError:
                #set_trace()
                thisflare = np.zeros(len(tt)) + 1e-6
            thisflare[thisflare < 1e-6] = 0.
            thisflare *= self.ydata.max()

        elif model == 'mendoza_constrained':
            param_i.add('tpeak0', value=X.params['tpeak' + str(nsol)].value)
            param_i.add('ampl0', value=X.params['ampl' + str(nsol)].value)
            param_i.add('fwhm0', value=X.params['fwhm' + str(nsol)].value)
            #try:
            thisflare = models.flare_model_mendoza(tt, param_i, complexity=1)
            #except ValueError:
            #    set_trace()

        return thisflare

    def get_bump_profile(self):
        if 'result_nodip' not in dir(self):
            print('You must run a fit first.')
            return

        return models.gaussian_bump(self.tdata.value, self.result_nodip.params)

    def get_full_profile(self, result, use_residuals=True, mode='mendoza_constrained'):
        '''
        The model without residuals has no polynomial added.
        '''
        if 'result_nodip' not in dir(self):
            print('You must run a fit first.')
            return

        if use_residuals:
            if mode == 'mendoza_constrained':
                return self.ydata - result.residual*self.yerrdata
            else:
                y = self.ydata/self.ydata.max()
                yerr = self.yerrdata/self.ydata.max()
                return (y - result.residual*yerr)*self.ydata.max()
        else:
            tt = (self.tdata/self.t12).value
            return models.exp_convolved(tt, result.params, self.npeaks) \
                + models.flare_dip(result.params, tt)

    def get_single_duration(self, n, double_t=True):
        '''
        Flare #n (in case of complex profile) duration based
        on pre-determined noise level.
        '''

        if double_t:
            tt = np.arange(self.tdata.min().value, self.tdata.max().value*2, \
                    np.diff(self.tdata).min().value)*u.d
        else:
            tt = np.copy(self.tdata)

        thisflare = self.get_single_profile(n, double_t=double_t)
        flare_start = tt[thisflare > self.noise_level][0]
        flare_end = tt[thisflare > self.noise_level][-1]

        return flare_end - flare_start

    def plot_models(self, plotname=None, plot_instance=None, \
            mode='mendoza_constrained', plot_LS=True, title='', \
            showplot=False):

        fitth = self.get_full_profile(self.result_nodip)

        #fig, axs = plt.subplots(figsize=(20, 10))
        left, width = 0.1, 0.85
        bottom, height = 0.5, 0.4

        if plot_LS:
            rect_flare = [left, bottom, width, height]
            rect_resid = [left, bottom - 0.1, width, 0.1]
            rect_periodo = [left, bottom - 0.4, width, 0.2]
            fig = plt.figure(figsize=(20, 10))
        else:
            rect_flare = [left, 0.35, width, 0.55]
            rect_resid = [left, 0.15, width, 0.2]
            #if 'dip_candidate' in plotname:
            fig = plt.figure(figsize=(13, 5))
            #elif 'QPP_candidate' in plotname:
            #    fig = plt.figure(figsize=(12, 5))

        axflare = plt.axes(rect_flare)
        axresid = plt.axes(rect_resid)
        if plot_LS:
            axperiodo = plt.axes(rect_periodo)

        tplot = self.tdata.to(u.min).value
        axflare.plot(tplot, self.ydata, 'k')

        for nsol in np.arange(self.npeaks):
            thisflare = self.get_single_profile(nsol, double_t=False)
            axflare.plot(tplot, thisflare)

        model_dip = self.get_full_profile(self.result_dip)
        axflare.plot(tplot, model_dip, linewidth=3, \
                label=r'Dip: $\Delta$AIC (No dip/dip) = {:.2f} ({:.1f} $\sigma$)'.format(self.dip_delta_aic, self.dip_significance))

        #model_cme = self.get_full_profile(self.result_cme)
        #axflare.plot(tplot, model_cme, linewidth=3, \
        #        label='CME: $\Delta$AIC (No CME/CME) = {:.2f} ({:.1f} $\sigma$)'.format(self.cme_delta_aic, self.cme_significance))

        if self.npeaks == 1:
            label = str(self.npeaks) + ' flare'
        else:
            label = str(self.npeaks) + ' flares'

        axflare.plot(tplot, fitth, linewidth=3, label='Flare profile')
        axflare.set_xticks([])
        if mode == 'mendoza_constrained':
            axflare.plot(tplot, np.polyval([self.result_nodip.params['d'], \
                self.result_nodip.params['e'], self.result_nodip.params['f']], \
                self.tdata.value), label='Quiet stellar flux')
        else:
            axflare.plot(tplot, np.polyval([self.result_nodip.params['d'], \
                self.result_nodip.params['e'], self.result_nodip.params['f']], \
                (self.tdata/self.t12).value)*self.ydata.max(), label='Quiet stellar flux')
        axflare.plot(self.traw*24.*60., self.yraw - np.nanmedian(self.yraw), 'k', \
                alpha=0.4, label='Scaled raw flux')
        axresid.set_xlabel('Time since peak [min]', fontsize=16)
        axflare.set_ylabel('Relative flux - 1', fontsize=16)
        try:
            if 'QPP_candidate' not in plotname:
                axflare.legend(loc='upper left', frameon=True, prop={'size': 14})
        except TypeError:
            set_trace()
        axresid.set_ylabel('Residuals', fontsize=16)

        if plot_LS:
            axperiodo.set_xlabel('Frequency [mHz]', fontsize=16)
            axperiodo.set_ylabel('Power', fontsize=16)
        if 'QPP_candidate' not in plotname:
            axflare.set_title(label + r' ($\tilde{\chi}^2=$' \
            + str(np.round(self.result_nodip.redchi, 2)) + ')', fontsize=18)
        else:
            axflare.set_title(title, fontsize=18)
        #axperiodo.set_xticks(fontsize=14)
        #axperiodo.set_yticks(fontsize=14)

        if plot_instance == 'dip':
            axresid.plot(tplot, self.result_dip.residual*self.yerrdata, 'k')
        else:
            axresid.plot(tplot, self.result_nodip.residual*self.yerrdata, 'k')
        if plot_LS:
            try:
                freq, power = self.ls_residuals.autopower(nyquist_factor=0.5)
                falev = self.ls_residuals.false_alarm_level(0.01)
                axperiodo.semilogx(freq.value*1e3, power)
                axperiodo.semilogx([freq[0].value*1e3, freq[-1].value*1e3], [falev, falev], \
                        'r--', label='1 % FAP')
                axperiodo.legend(loc='upper right', prop={'size': 14})
            except FloatingPointError:
                pass

        plt.tight_layout()
        if plot_instance is not None and '_dip_candidate' not in plotname \
                    and '_QPP_candidate' not in plotname:
            plot_instance.savefig(fig)
        elif plot_instance is not None and ('dip_candidate' in plotname \
                or 'QPP_candidate' in plotname):
            plt.savefig(plotname)
        if showplot:
            plt.show()
            set_trace()

        return

    def flare_rate(self, t, mu, Escale, alpha):
        '''
        Defines flare rate with law nu = mu*E**{-alpha) (e.g. Loyd+2018)

        Parameters:
        -----------
        en (numpy array, normalized): minimum to maximum flare energy

        Returns:
        --------
        tpeak_distr (dictionary): A number of peak times/energies with both
                                  frequency and energy depending on the law
        '''

        tpeak_distr = {}
        # Define a range of energies (normalized)
        for Ei in np.arange(Escale, Escale*5, Escale):
            nu = mu*(Ei/Escale)**(-alpha)*24*60 # [1/min]
            print('Flares with E =', Ei, 'J, rate =', nu, '/min')
            # No of flares = uniform distribution on t-axis/rate
            tpeak_i = np.random.uniform(low=t.min(), high=t.max(),
                        size=int((t.max() - t.min())*nu))
            tpeak_distr[Ei] = tpeak_i

        return tpeak_distr

    def thresholding_algo(self, y, lag, threshold, influence):
        '''
        Implementation of algorithm from
        https://stackoverflow.com/a/22640362/6029703
        '''
        signals = np.zeros(len(y))
        filteredY = np.array(y)
        avgFilter = [0]*len(y)
        stdFilter = [0]*len(y)
        avgFilter[lag - 1] = np.mean(y[0:lag])
        stdFilter[lag - 1] = np.std(y[0:lag], ddof=1)
        for i in range(lag, len(y)):
            if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
                if y[i] > avgFilter[i-1]:
                    signals[i] = 1
                else:
                    signals[i] = -1

                filteredY[i] = influence * y[i] + \
                               (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

        return dict(signals = np.asarray(signals),
                    avgFilter = np.asarray(avgFilter),
                    stdFilter = np.asarray(stdFilter))

    def get_flare_energy(self, method='shibayama', double_t=True, npeak=-1):
        '''
        Given a flare profile fit and temperature, computes total energy output.
        Formulae from Gunther+2020.
        See also https://arxiv.org/pdf/1810.03277.pdf

        The Equivalebt Duration is computed also for the fast rise-fast decay
        and for the gradual decay phase.

        Parameters
        ----------
        tprof [array]: must be in seconds
        wth, fth: filter throughput (wth in A)
        rms: if >0, only the flux above the rms level is considered for energy
        determination.
        double_t: whether to double the time time axis duration to avoid
        cutting flare profiles. If True, the time axis must be measured in days.
        '''

        if double_t:
            tt = np.arange(self.tdata.min().value, self.tdata.max().value*2, \
                    np.diff(self.tdata).min().value)*u.d
        else:
            tt = np.copy(self.tdata)

        if method == 'shibayama':
            bbstar = BlackBody(self.Tstar)(self.wth*u.AA)
            bbflare = BlackBody(self.Tflare)(self.wth*u.AA)
            lumratio = trapz(bbstar*self.fth, x=self.wth) \
                        / trapz(bbflare*self.fth, x=self.wth)
            Aflare_t = self.yprof*np.pi*self.Rstar**2*lumratio
            Lflare = constants.sigma_sb*(self.Tflare**4)*Aflare_t
            self.flare_energy = trapz(Lflare, x=self.tdata.to(u.s)).to(u.erg)

        elif method == 'davenport':
            if npeak > -1:
                #tfast = tt.to(u.s) <= (self.t12.value*u.day).to(u.s)
                fwhm = self.result_nodip.params['fwhm' + str(npeak)].value*u.day
                tfast = tt.to(u.s) <= fwhm.to(u.s)
                self.EDfast = np.trapz(self.yprof[tfast], x=tt[tfast].to(u.s))
                self.EDslow = np.trapz(self.yprof[~tfast], x=tt[~tfast].to(u.s))
            ED = np.trapz(self.yprof, x=tt.to(u.s))
            if 'flare_luminosity' in dir(self):
                self.flare_energy = ED*self.flare_luminosity
            else:
                print('You must compute the flare luminosity first.')
                set_trace()

        return self.flare_energy

    def get_flare_luminosity(self, instrument, mag, distance, flare_amplitude):
        '''
        Use stellar magnitude, distance, and instrument zero point (from SVO service)
        to convert magnitude to quiescent stellar luminosity. Vega mag is assumed
        to be 0.
        '''
        if instrument == 'CHEOPS': # using Gaia G bandpass
            zeropoint = 2.49769e-9
            lambdaeff = 5850.88
        if instrument == 'TESS':
            zeropoint = 1.33161e-9
            lambdaeff = 7452.64

        # Flux densities --> fluxes with effective wavelength
        F0 = zeropoint*u.erg/u.cm**2/u.s/u.A
        F = F0*10**(-mag/2.5)*lambdaeff*u.A
        L_quiesc = 4.*np.pi*distance.to(u.cm)**2*F
        self.flare_luminosity = L_quiesc*flare_amplitude

        return self.flare_luminosity

    def dip_fit_davenport(self, x, y, yerr, complexity, flares_par, tbeg, \
                correlated_noise_pattern, plots=False, fit_continuum=True):
        '''
        Fit pre-flar dip with an asymmetric generalised Gaussian and previously
        found number of flares.
        Par for pre-dip is defined as [A, x0, w, n] (flat for n > 2,
        peaked for n < 2).

        tbeg: beginning time of the flare, i.e. where flare model without dip
        is < 1 ppm.
        flares_par contains 'noiselevel': used to include or reject Gausian bump fits
        correlated_noise_pattern: list of [bin, red noise, white noise] used
        to compute the significance of a dip with a given width
        '''

        print('This is old stuff!')

        initpars = flares_par['fit']
        noiselevel = flares_par['noiselevel']
        if initpars['g_ampl'] <= 3.*noiselevel:
            initpars['g_ampl'].value = 0.
            initpars['g_ampl'].vary = False
            initpars['g_t0'].vary = False
            initpars['g_w1'].vary = False
            initpars['g_w2'].vary = False
            initpars['g_n'].vary = False
        if not fit_continuum:
            initpars['d'].vary = False
            initpars['e'].vary = False
            initpars['f'].vary = False

        # Search for potential dip points (recompute residuals)
        initpars.add('A', value=0., vary=False)
        initpars.add('x0', value=-10., vary=False)
        initpars.add('w2', value=20./(24.*60), min=1./(24.*60), vary=False)
        initpars.add('w1', value=20./(24.*60), min=1./(24.*60), vary=False)
        initpars.add('n', value=2.1, vary=False)

        for nf in np.arange(complexity):
            initpars['t0' + str(nf)].vary=False

        result_nodip = minimize(gauss_residuals, initpars, args=(x, y, yerr, \
                complexity), method='powell', calc_covar=False)

        # Fix all pars and get only the continuum fit - this is useful for the
        # AIC comparison
        for p in initpars.keys():
            if p != 'f':
                result_nodip.params[p].vary = False
        result_nodip = minimize(gauss_residuals, result_nodip.params, \
            args=(x, y, yerr, complexity), method='powell', calc_covar=False)

        residuals_nodip = result_nodip.residual*yerr
        self.nodip = result_nodip
        model_nodip = asymmetric_gauss(result_nodip.params, x, complexity)

        # Periodogram of fit residuals (not normalized)
        ls = timeseries.LombScargle(x*u.day.to(u.s), residuals_nodip, \
                        normalization='standard', fit_mean=True)

        try:
            labels, anomaly = dip_classification(x, y, tbeg, plots=False)
        except FloatingPointError:
            labels = np.random.randint(low=0, high=2, size=np.sum(x < tbeg))
            anomaly = np.array([np.sum(labels == 0), np.sum(labels == 1)]).argmin()
            anomaly = [x[x < tbeg][labels == anomaly], \
                                            y[x < tbeg][labels == anomaly]]
        except ValueError:
            labels = np.random.randint(low=0, high=2, size=np.sum(x < tbeg))
            anomaly = np.array([np.sum(labels == 0), np.sum(labels == 1)]).argmin()
            anomaly = [x[x < tbeg][labels == anomaly], \
                                            y[x < tbeg][labels == anomaly]]

        if len(labels) > 0:
            self.labels = labels
            self.labelled_percentage = len(anomaly[0])/np.sum(x < tbeg)*100.
        elif len(labels) == 0 or np.sum(x < tbeg) == 0:
            self.dip = copy.deepcopy(result_nodip)
            self.dip_significance = -np.inf
            self.rms = np.std(residuals_nodip)
            self.labelled_percentage = 100.
            model_dip = np.copy(model_nodip)
            return model_dip, model_nodip, ls

        pars = Parameters()
        pars.add('A', value=-0.8, min=-1, max=0.)
        pars.add('x0', value=np.median(anomaly[0]), \
                min=max([np.median(anomaly[0]) - 10./(24*60.), x.min()]), \
                max=min([np.median(anomaly[0]) + 10./(24*60.), 0.]))
        pars.add('w1', value=3./(24.*60), min=0.1/(24.*60), \
                            max=20./(24.*60))
        pars.add('w2', value=3./(24.*60), min=0.1/(24.*60), \
                max=20./(24.*60))#np.ptp(anomaly[0]))
        pars.add('n', value=2.1, min=2., max=4.)

        for nf in np.arange(complexity):
            pars.add('t0' + str(nf), value=result_nodip.params['t0' + str(nf)].value, vary=False)
            pars.add('t12' + str(nf) + '_diff', \
                value=result_nodip.params['t12' + str(nf) + '_diff'], \
                min=result_nodip.params['t12' + str(nf) + '_diff'].min, \
                max=result_nodip.params['t12' + str(nf) + '_diff'].max)
            # t12 must be > t0, so an additional parameter is fitted instead
            pars.add('t12' + str(nf), expr='t0' + str(nf) \
                        + ' + t12' + str(nf) + '_diff')
            pars.add('A' + str(nf), value=result_nodip.params['A' + str(nf)], \
               min=result_nodip.params['A' + str(nf)].min, \
               max=result_nodip.params['A' + str(nf)].max)
            pars.add('tau1' + str(nf), value=result_nodip.params['tau1' + str(nf)], \
                    min=result_nodip.params['tau1' + str(nf)].min, \
                    max=result_nodip.params['tau1' + str(nf)].max)
            # Require tau1 < tau2
            pars.add('taumult' + str(nf), value=result_nodip.params['taumult' + str(nf)], \
                        min=result_nodip.params['taumult' + str(nf)].min, \
                        max=result_nodip.params['taumult' + str(nf)].max)
            pars.add('tau2' + str(nf),  \
                    expr='tau1' + str(nf) + ' * taumult' + str(nf))
                    #max=x.max()/3.)
            pars.add('taurise_mult' + str(nf), \
                        value=result_nodip.params['taurise_mult' + str(nf)], \
                        min=result_nodip.params['taurise_mult' + str(nf)].min, \
                        max=result_nodip.params['taurise_mult' + str(nf)].max)
            pars.add('taurise' + str(nf), expr='tau1' + str(nf) \
                            + ' * taurise_mult' + str(nf))
            #pars.add('taurise' + str(nf), value=initpars['taurise' + str(nf)], \
            #        min=initpars['taurise' + str(nf)].min9, \
            #        max=initpars['taurise' + str(nf)]*1.01 + 1e-5)

        pars.add('d', value=result_nodip.params['d'])#, min=initpars['d'].min9, \
        #                max=initpars['d']*1.01, vary=True)
        pars.add('e', value=result_nodip.params['e'])#, min=initpars['e'].min9, \
        #                    max=initpars['e']*1.01, vary=True)
        pars.add('f', value=result_nodip.params['f'])#, min=initpars['f'].min9, \
        #                    max=initpars['f']*1.01, vary=True)
        pars.add('g_ampl', value=result_nodip.params['g_ampl'], \
                        min=result_nodip.params['g_ampl'].min, \
                        max=result_nodip.params['g_ampl'].max)
        pars.add('g_t0', value=result_nodip.params['g_t0'], \
                            min=result_nodip.params['g_t0'].min, \
                            max=result_nodip.params['g_t0'].max)
        pars.add('g_w1', value=result_nodip.params['g_w1'], \
                            min=result_nodip.params['g_w1'].min, \
                            max=result_nodip.params['g_w1'].max)
        pars.add('g_w2', value=result_nodip.params['g_w2'], \
                            min=result_nodip.params['g_w2'].min, \
                            max=result_nodip.params['g_w2'].max)
        pars.add('g_n', value=result_nodip.params['g_n'], \
                            min=result_nodip.params['g_n'].min, \
                            max=result_nodip.params['g_n'].max)
        for p in pars:
            if p != 'f' and p != 'A' and p != 'x0' \
            and p != 'w1' and p != 'w2' and p != 'n':
                pars[p].vary = False

        # Fit with dip
        result_dip = minimize(gauss_residuals, pars, args=(x, y, yerr, \
                    complexity), method='powell', calc_covar=False)
                    #float_behavior='chi2')
        residuals_dip = result_dip.residual*yerr
        model_dip = asymmetric_gauss(result_dip.params, x, complexity)
        self.dip = result_dip

        # Compute significance on the basis of correlated noise level (see Pont+2006)
        dip_width = (result_dip.params['w1'] + result_dip.params['w2']) \
                    /np.diff(x).min()
        rednoise_level = correlated_noise_pattern[1][ \
            abs(np.array(correlated_noise_pattern[0]) - dip_width).argmin()]

        self.dip_significance = abs(result_dip.params['A'].value) \
                        /(rednoise_level/1e6)

        #self.dip_significance \
        #            = result_dip.params['A'].value/np.std(residuals_dip)
        self.rms = np.std(residuals_dip)

        # KS test for difference between the two residual distributions before
        # the beginning of the flare
        self.kstest = ks_2samp(residuals_dip[x < tbeg], \
                                            residuals_nodip[x < tbeg])

        if plots:
            fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
            axs[0].errorbar(x*24.*60., y, yerr=yerr, fmt='.', alpha=0.1)
            axs[0].plot(x*24.*60., model_dip, linewidth=3, \
                                            label='LM model with dip')
            axs[0].plot(x*24.*60., model_nodip, linewidth=3, \
                                            label='LM model without dip')
            axs[0].plot(flares_par['Total_fit'][0]*24.*60., \
                flares_par['Total_fit'][1], linewidth=3, label='Initial model')
            axs[0].legend()
            freq = np.linspace(1e-4, 1e-2, 1000)
            pgram = ls.power(freq)
            axs[1].loglog(freq*1e6, pgram)
            axs[1].set_xlabel(r'Frequency [$\mu$Hz]', fontsize=14)
            axs[1].set_ylabel('Power')
            plt.show()
            set_trace()

        return model_dip, model_nodip, ls#, freq, PSD

def flare_oscillations(par, x, data, err, complexity, plots=False):
    '''
    Flare profile with damped oscillations pre and post flare
    '''

    # Add flare
    y = models.exp_line(x, par, complexity=complexity)

    # The central point for oscullations corresponds to the fitst flare peak
    t0 = []
    for nf in np.arange(complexity):
        t0.append(par['t0' + str(nf)])
    x0 = min(t0)
    y[x < x0] += par['B1']*np.sin(2.*np.pi*x[x < x0]/par['P1'] + par['x01']) \
                *np.exp(-abs(x[x < x0] - x0)*par['m1'])
    y[x >= x0] += par['B2']*np.sin(2.*np.pi*x[x >= x0]/par['P2'] + par['x02']) \
                *np.exp(-abs(x[x >= x0] - x0)*par['m2'])

    if np.sum(np.isinf(y)) > 0:
        set_trace()

    if plots:
        plt.errorbar(x, data, yerr=err, fmt='.')
        plt.plot(x, y)
        plt.show()
        set_trace()

    return (data - y)/err

# Define a cost function
def neg_log_like(params, y, gp):
    parvalues = []
    for p in ['log_S0', 'log_Q', 'log_omega0']:#, 'log_white_noise']:
        parvalues.append(params[p].value)
    parvalues = np.array(parvalues)

    gp.set_parameter_vector(parvalues)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]

def dip_classification(xflare, yflare, tbeg, plots=False):
    '''
    Gaussian mixture model to classify time series points based on their
    value, first and second derivative.
    '''

    x = xflare[xflare < tbeg]
    y = yflare[xflare < tbeg]

    if len(x) < 3:
        return np.zeros(len(x)), [x, y]

    y2 = np.diff(y)[:-1]
    #y3 = np.diff(np.diff(y))
    y1 = y[:-2]
    y3 = y1 - signal.medfilt(y1, kernel_size=31)
    y4 = y2 - signal.medfilt(y2, kernel_size=31)
    X = preprocessing.scale(np.transpose([y1, y2, y3, y4]))

    clustering = mixture.GaussianMixture(n_components=2, n_init=10)
    #clustering = cluster.KMeans(n_clusters=2)
    labels = clustering.fit_predict(X)
    anomaly = np.array([np.sum(labels == 0), np.sum(labels == 1)]).argmin()
    anomaly = [x[:-2][labels == anomaly], y1[labels == anomaly]]

    if plots:
        plt.figure()
        plt.plot(xflare[xflare >= tbeg], yflare[xflare >= tbeg], 'r.', alpha=0.5)
        #for j in np.arange(2):
            #plt.plot(x[:-2][labels == j], y1[labels == j])
        plt.scatter(x[:-2], y1, marker='.', c=clustering.predict_proba(X)[:, 0])
        plt.colorbar()

        fig, axs = plt.subplots(ncols=6, figsize=(24, 5))
        for j in np.arange(2):
            axs[0].scatter(X[:,0][labels == j], X[:,1][labels == j])
            axs[1].scatter(X[:,0][labels == j], X[:,2][labels == j])
            axs[2].scatter(X[:,0][labels == j], X[:,3][labels == j])
            axs[3].scatter(X[:,1][labels == j], X[:,2][labels == j])
            axs[4].scatter(X[:,1][labels == j], X[:,3][labels == j])
            axs[5].scatter(X[:,3][labels == j], X[:,3][labels == j])

        plt.show()
        set_trace()
        plt.close('all')

    return labels, anomaly
