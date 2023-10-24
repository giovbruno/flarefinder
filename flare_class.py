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
                fit_params.add('fwhm' + str(nf) , \
                        min=np.diff(tt).min()*2., max=self.tend - self.tbeg)
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

            # Additional peaks? If yes, continue
                #newpeaks = list(peakut.indexes(result.residual*self.yerrdata, \
                #        thres=threshold*np.std(result.residual*self.yerrdata), \
                #        min_dist=10, thres_abs=True))
                #if len(newpeaks) == 0:
                #    print('No more peaks')
                #    break

            if n == complexity:
                break

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
            model = models.flare_model_mendoza(tt, X.params, \
                    complexity=self.npeaks, plots=False)
            fig, axs = plt.subplots(nrows=2, sharex=True)
            axs[0].plot(tt, self.ydata, 'k', alpha=1.)
            model = self.ydata - X.residual*self.yerrdata
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

    def get_single_profile(self, nsol, model='mendoza_constrained'):
        '''
        Compute a model for a single flare component.

        Parameters
        ----------
        nsol: number of flare compoennt (1 for single-peak flares)
        '''
        
        if 'result_nodip' not in dir(self):
            print('You must run a fit first.')
            return

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
            thisflare = models.flare_model_mendoza(self.tdata.value, param_i, \
                    complexity=1)
            #except FloatingPointError:
            #    set_trace()
        return thisflare

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

    def get_single_duration(self, n):
        '''
        Flare #n (in case of complex profile) duration based
        on pre-determined noise level.
        '''
        thisflare = self.get_single_profile(n)
        flare_start = self.tdata[thisflare > self.noise_level][0]
        flare_end = self.tdata[thisflare > self.noise_level][-1]

        return flare_end - flare_start

    def plot_models(self, plotname=None, plot_instance=None, \
            mode='mendoza_constrained', plot_LS=True, title='', \
            showplot=False):
        '''
        Plot all components in a single or complex flare profile.
        '''
                
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
            fig = plt.figure(figsize=(12, 5))
            #elif 'QPP_candidate' in plotname:
            #    fig = plt.figure(figsize=(12, 5))

        axflare = plt.axes(rect_flare)
        axresid = plt.axes(rect_resid)
        if plot_LS:
            axperiodo = plt.axes(rect_periodo)

        tplot = self.tdata.to(u.min).value
        axflare.plot(tplot, self.ydata, 'k')

        for nsol in np.arange(self.npeaks):
            thisflare = self.get_single_profile(nsol)
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
        axresid.set_xlabel('Time since peak [min]', fontsize=18)
        axflare.set_ylabel('Relative flux - 1', fontsize=18)
        try:
            if 'QPP_candidate' not in plotname:
                axflare.legend(frameon=False, prop={'size': 14})
        except TypeError:
            set_trace()
        axresid.set_ylabel('Residuals', fontsize=18)

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

    def get_flare_energy(self, method='shibayama'):
        '''
        Given a flare profile fit and temperature, computes total energy output.
        Formulae from Gunther+2020.
        See also https://arxiv.org/pdf/1810.03277.pdf

        Parameters
        ----------
        tprof [array]: must be in seconds
        wth, fth: filter throughput (wth in A)
        rms: if >0, only the flux above the rms level is considered for energy
        determination.
        '''

        if method == 'shibayama':
            bbstar = BlackBody(self.Tstar)(self.wth*u.AA)
            bbflare = BlackBody(self.Tflare)(self.wth*u.AA)
            lumratio = trapz(bbstar*self.fth, x=self.wth) \
                        / trapz(bbflare*self.fth, x=self.wth)
            Aflare_t = self.yprof*np.pi*self.Rstar**2*lumratio
            Lflare = constants.sigma_sb*(self.Tflare**4)*Aflare_t
            self.flare_energy = trapz(Lflare, x=self.tdata.to(u.s)).to(u.erg)

        elif method == 'davenport':
            ED = np.trapz(self.yprof, x=self.tdata.to(u.s))
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
