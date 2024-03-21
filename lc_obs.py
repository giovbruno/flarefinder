#! /usr/bin/env python3

'''
Creates a light curve object with operations to work on it
'''

import numpy as np
import numpy.ma as ma
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import timeseries as ats
import astropy.units as u
from astropy import stats as astats
from scipy import signal, stats
import sys
#sys.path.append('/home/giovanni/Shelf/python/peakutils/')
import peakutils
import pandas as pd
import pickle
import pycheops
from dace_query.cheops import Cheops
from lmfit import minimize, Parameters, fit_report
import smooth
from pdb import set_trace
import astropy.units as u
muHz = u.def_unit('muHz', 1e-6*u.Hz)
import phot_analysis
from astropy import timeseries
import harvey
import corner

class LC:
    def __init__(self, lctype='', lc_file=None, system_pars={}, \
                t=None, y=None, yerr=None, aperture='OPTIMAL'):
        '''
        lctype: Kepler, CoRoT, file ASCII, modeled or arrays.
                Reference JD for CoRoT @ Jan 1, 2000, 12:00:00
                (CoRoT: manual N2 data + https://ssd.jpl.nasa.gov/tc.cgi#top)
        '''
        self.lc_file = lc_file
        self.system_pars = system_pars
        self.lctype = lctype
        #if np.logical_and.reduce((type(t) == np.ndarray, \
        #            type(y) == np.ndarray, type(yerr) == np.ndarray)):
        self.t = t
        self.y = y
        self.yerr = yerr
        self.aperture = aperture

    def prepare_lc(self, norm_flux_max=False, norm_flux_median=False, \
                    clip=False, isplot=False, remove_nan=False):
        '''
        Reads LC file and cleans from crowding (for Kepler), just take the
        relevant fields (for CoRoT), or just read the modeled LC

        Input
        -----
        lctype (string): whether Kepler, CoRoT, TESS, CHEOPS or else
                            (case insensitive)

        Returns
        ------
        time, flux and uncertainties arrays
        '''

        if self.lctype.lower() == 'kepler':
            lc = fits.open(self.lc_file)
            # LC parameters
            crowding = lc[1].header['CROWDSAP']
            #print(self.lc_file, 'Crowding:', crowding)
            flfrcsap = lc[1].header['FLFRCSAP']
            quarter = lc[0].header['QUARTER']
            season = lc[0].header['SEASON']
            t_begin = float(lc[1].header['TUNIT1'][6:])
            # Data
            t = t_begin + lc[1].data['TIME']
            y_raw = lc[1].data['SAP_FLUX']
            # Error on flux is scaled
            yerr = lc[1].data['SAP_FLUX_ERR']*crowding

        elif self.lctype.lower() == 'corot':
            lc = fits.open(self.lc_file)
            # LC parameters
            status = lc[1].data['STATUS']
            dev = lc[1].data['REDFLUXDEV']
            # Good and bad pixels
            goodpix = status == 0
            # Sampling
            thirtytwo = dev == 0
            # Data
            tref = 2451545.
            t = lc[1].data['DATEJD'][np.logical_and(goodpix, thirtytwo)] \
                    + tref
            y_raw = lc[1].data['WHITEFLUX'][np.logical_and(goodpix, thirtytwo)]
            yerr = y_raw**0.5

        elif self.lctype.lower() == 'tess':
            data = fits.open(self.lc_file)[1].data
            t = data['TIME']
            y_raw = data['PDCSAP_FLUX']
            yerr = data['PDCSAP_FLUX_ERR']
            quality = data['QUALITY']
            header = fits.open(self.lc_file)[0].header

        elif self.lctype.lower() == 'cheops':
            try:
                data = pycheops.Dataset(self.lc_file, download_all=True, \
                    verbose=False, view_report_on_download=False)
            except OSError:
                set_trace()
            except:
                print(self.lc_file, ': general exception')
            t, y_raw, yerr = data.get_lightcurve(self.aperture, \
                    decontaminate=False)
            table = data.get_lightcurve(self.aperture, \
                        decontaminate=False, returnTable=True)
            print('Using ', self.aperture, 'aperture.')

        elif self.lctype.lower() == 'generic_fits':
            t, y_raw, yerr = pickle.load(open(self.lc_file, 'rb'))

        elif self.lctype.lower() == 'generic_ascii':
            #t, y_raw, yerr = np.loadtxt(self.lc_file, unpack=True,
            #                        comments=('\\', '|'), usecols=(0, 1, 2))
            data = pd.read_csv(self.lc_file, usecols=(1, 25, 26, 27, 34), \
                comment='#', names=['bjd', 'y_raw', 'y_err', 'quality', 'y_fit'])
            t = data['bjd']
            q = data['quality']
            y = data['y_fit']
            y_err = data['y_err']
            flag = q == 'G'
            t, y_raw, yerr = t[flag], y[flag], y_err[flag]
        else:
            t, y_raw, yerr = self.t, self.y_raw, self.yerr

        # Remove NaNs before taking medians ecc
        if remove_nan:
            flag_nan = np.logical_or(np.isnan(y_raw), np.isnan(yerr))
            #if np.sum(flag_nan) > 0:
                #print('NaN in LC')
            t, y_raw, yerr = t[~flag_nan], y_raw[~flag_nan], yerr[~flag_nan]

        if self.lctype.lower() == 'tess':
            quality = quality[~flag_nan]

        # Remove contamination
        if self.lctype.lower() == 'kepler':
            y = y_raw - np.median(y_raw*(1. - crowding))
        #elif self.lctype.lower() == 'corot' or self.lctype.lower() == 'generic':
        else:
            y = np.copy(y_raw)

        # Normalize flux values?
        if norm_flux_max:
            yerr /= y.max()
            y /= y.max()
        elif norm_flux_median:
            yerr /= np.median(y)
            y /= np.median(y)

        # Sigma clipping?
        if clip:
            yfilt = signal.medfilt(y, 11)
            yclip = astats.sigma_clip(y - yfilt, sigma=5)
            t, y, yerr = ma.array(t), ma.array(y), ma.array(yerr)
            if self.lctype.lower() == 'tess':
                quality = ma.array(quality)
            t.mask = yclip.mask
            y.mask = yclip.mask
            yerr.mask = yclip.mask
            if self.lctype == 'tess':
                quality.mask = yclip.mask

        if isplot:
            plt.errorbar(t, y, yerr=yerr, fmt='.')
            plt.xlabel('Time [BJD]', fontsize=16)
            if norm_flux_max or norm_flux_median:
                plt.ylabel('Normalized flux', fontsize=16)
            else:
                plt.ylabel('Flux [e$^-$/s]', fontsize=16)
            plt.show()

        if self.lctype.lower() == 'tess':
            return t, y, yerr, quality, header
        elif self.lctype.lower() == 'cheops':
            return table, data.lc['time'] + data.lc['bjd_ref'], y, yerr, data
        else:
            return t, y, yerr

    def remove_transits(self, t, y, yerr, isplot=0):
        '''
        Removes transits from the light curve using time of initial transit
        and planet orbital period.

        Input
        -----
        t, y, yerr arrays from prepare_lc

        Returns
        -------
        time, flux and error arrays with transits removed
        '''

        P = self.system_pars['Pplanet']
        td = self.system_pars['transit_duration']
        ti = self.system_pars['first_transit'] #+ self.tref
        tti = ti
        i = 0
        while tti < t.max():
            tti = ti + i*P
            flag = np.logical_and(t > tti - td*1.0, t < tti + td*0.5)
            t, y, yerr = t[~flag], y[~flag], yerr[~flag]
            i += 1

        if isplot != 0:
            plt.plot(t, y, '.')
            plt.show()
            set_trace()

        return t, y, yerr

    def autocorr(self, t, y, detrend=True, filter=1, isplot=0):
        '''
        Sep 6, 2019: autocorrelation function (from McQuillan+2013).

        Input
        -----

        Filter: Box width to median-filter the data (1 means no filtering)
        '''

        if detrend:
            y = signal.detrend(y)
        if filter > 1:
            medy = signal.medfilt(y, filter)
            print('Applied median filter with', filter, 'time-units window.')
        # Define frequency
        nyq = 0.5*np.diff(t).min()  # Nyquist frequency
        #f = np.arange(0.01, 10, nyq)
        #tt = 2*np.pi/f
        #pgram = signal.lombscargle(t, y, f, normalize=True, precenter=True)
        norm = np.sum(y**2)
        set_trace()
        #pgram = np.correlate(y, y, mode='full')/norm
        lags, pgram = plt.acorr(y, maxlags=100, usevlines=False)[:2]
        lags, pgram = lags[lags >= 0.], pgram[lags >= 0.]
        f = lags*np.mean(np.diff(t))
        maxp = peakutils.indexes(pgram)[0]
        # Find periodogram maximum
        #maxp = pgram.argmax()
        fmax = f[maxp]
        #tmax = tt[maxp]
        print('Maximum peak:', fmax, 'days.')
        if isplot != 0:
            plt.close('all')
            fig = plt.figure(33, figsize=(12, 8))
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(t, y, 'k.')
            ax1.set_xlabel('Time [days]', fontsize=16)
            ax1.set_ylabel('Normalized flux', fontsize=16)
            ax1.set_title('Time series', fontsize=16)
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(f, pgram, 'k')
            #ax2.set_xlabel(r'Log frequency [$2 \pi \times$ day$^{-1}$]', fontsize=16)
            ax2.set_xlabel('Period [days]', fontsize=16)
            #ax2.set_ylabel('LS periodogram power', fontsize=16)
            ax2.set_ylabel('ACF', fontsize=16)
            for i in np.arange(1, 4):
                if i == 1:
                    ax2.plot([i*fmax, i*fmax], [0, pgram.max()], 'r--', \
                        label='Max peak and harmonics')
                    ax2.legend(loc='best', frameon=False, fontsize=16)
                else:
                    ax2.plot([i*fmax, i*fmax], [0, pgram.max()], 'r--')
            plt.show()

        return fmax, pgram.max()

    def variance_P(self, binw=0.05, synodic=0, isplot=2):
        '''
        Measures flux variance as a function of planetary orbital phase

        Paramters
        ---------
        binw:   width of the bin in planetary phase
        synodic: whether to use the planetary phase (0) or the synodic
                 period wrt stellar rotation period (!= 0)
        Returns
        -------
        planetary phase array, flux variance array
        NB. For some LC chunks, not all phases are present
        '''

        t, y, yerr = self.prepare_lc(norm_flux=0, clip=1, isplot=0)
        t, y, yerr = self.remove_transits(t, y, yerr, isplot=0)

        # First 75 days of obs
        flag = t < t.min() + 75.
        t, y, yerr = t[flag], y[flag], yerr[flag]
        #y -= signal.medfilt(y, 31)

        # Divide phase in chunks of 0.05
        Ppl = self.system_pars['Pplanet']
        ti = self.system_pars['first_transit']
        if synodic == 0:
            phase = ((t - ti) % Ppl)/Ppl
        else:
            Pstar = self.system_pars['Pstar']
            Psyn = (1./Ppl - 1./Pstar)**-1
            phase = (t - ti) % Psyn/Psyn
        binphase = np.arange(0.05, 1., binw)
        # Arrays to save results
        tp = np.zeros(len(binphase) - 1)
        yp = np.copy(tp)
        for i in np.arange(1, len(binphase)):
            phase_in = np.logical_and(phase >= binphase[i-1],
                                    phase < binphase[i])
            tph, yph, errph = phase[phase_in], y[phase_in], yerr[phase_in]
            # Variance wrt median filter of the segment
            #filt = signal.medfilt(yph, 11)
            yph = astats.sigma_clip(yph, sigma=5)
            vari = np.var(yph, ddof=1)
            #vari = self.pooled_variance2(tph, yph, \
            #    multfact=int((t.max() - t.min())/(self.system_pars['Pplanet'])))
            #meani = np.mean(yph - filt)
            #tp_bin = np.mean(tph)
            tp_bin = np.mean([binphase[i - 1], binphase[i]])
            #Save results
            tp[i-1] = np.mean([binphase[i - 1], binphase[i]])
            yp[i-1] = vari

            if isplot == 2:
                plt.figure(1)
                plt.plot(tp_bin, vari, 'ko')
                #plt.plot(tph, yph, 'o')
                if synodic == 1:
                    plt.xlabel('Synodic phase', fontsize=16)
                else:
                    plt.xlabel('Orbital phase', fontsize=16)
                plt.ylabel('Flux variance per phase bin (' + str(binw)
                            +' wide)', fontsize=16)

                #plt.figure(2)
                #plt.plot(tp_bin, meani, 'ko')
                #plt.title('Mean')
                #plt.show()

        # Remove NaN
        return tp, yp#[~np.isnan(yp)], yp[~np.isnan(yp)]

    def pooled_variance_time(self, pplanet=False, \
                remove_transits=False, newy=[0], adapt_bins=True, \
                detrend=True, mad=False, plots=False, RMS=False, bins=[]):
        '''
        Calculates the pooled variance (NOT standard deviation)
        of a series of observations.

        Input
        -----
        yarr: arrays containing as many entries as the population samples.
        multfact: needed for variance_P
        See https://en.wikipedia.org/wiki/Pooled_variance

        pplanet (bool): whether to multiply bins by the planet period
        remove_transits (bool): are there transits to remove?
        newy: to provide a corrected flux array
        adapt_bins: resample binning for unevenly sampled datasets
        RMS: instead of pooled variance, just compute RMS wrt bins
        bins: if not provided, a default set of bins will be used
        '''

        if self.lctype.lower() == 'cheops':
            t_, t, y, yerr, _ = self.prepare_lc(norm_flux_median=1, clip=1, isplot=0)
        #elif self.lctype.lower() == 'cheops_imagette':
        #    t, y = newy[0], newy[1]
        #elif self.lctype.lower() == 'array':
        #    set_trace()
        #    t, y, yerr = self.t, self.y, self.yerr
        elif self.lctype.lower() != 'custom':
            t, y, yerr = self.prepare_lc(norm_flux_median=1, clip=1, isplot=0)

        if len(newy) > 1:
                t, y = newy[0], newy[1]
        if remove_transits:
            t, y, yerr = self.remove_transits(t, y, yerr, isplot=0)
        #y -= signal.medfilt(y, 31)
        # Total duration of LC
        ttot = (t.max() - t.min())#*multfact
        # Time span of every bin
        if pplanet:
            bins = bins*self.system_pars['Pplanet']
        if detrend:
            y = signal.detrend(y)
        # Compare bins with largest non-interrupted chunk
        #tdiff = np.diff(t)
        #interr = np.where(tdiff > 10*abs(tdiff).min())
        #lags = np.diff(np.concatenate(([1], interr[0], [len(tdiff) - 1])))
        #bins = np.linspace(bins[0], lags.max()*abs(np.diff(t)).min(), len(bins))
        #set_trace()
        # The bins have to be shorter than 1/3rd of the LC
        #bins = bins[bins < t.max()/3.]
        # for each LC binning
        #set_trace()
        deltat = np.diff(t).min()*3
        # Check
        if deltat <= 0:
            deltat = 60./86400
        interr = np.where(np.diff(t) > 5.*deltat)[0]
        if len(interr) > 0:
            # This should indicate some issue
            if interr[0] == 0:
                interr = interr[1:]
            interv = np.concatenate(([0], interr + 1, [len(t) - 1]))
            intervdiff = np.diff(interv)
            if intervdiff.argmax() == 0:
                tchunk = t[0:intervdiff.max() - 1]
                ychunk = y[0:intervdiff.max() - 1]
            elif intervdiff.argmax() == len(intervdiff) - 1:
                ini = interv[intervdiff.argmax()] + 1
                tchunk = t[ini:-1]
                ychunk = y[ini:-1]
            else:
                ini = interv[intervdiff.argmax()] + 1
                fin = interv[intervdiff.argmax() + 1] - 1
                tchunk = t[ini:fin]
                ychunk = y[ini:fin]
        else:
            tchunk = t.copy()
            ychunk = y.copy()
        try:
            lentchunk = tchunk.max() - tchunk.min()
        except ValueError:
            set_trace()

        #for j, gap in enumerate(interr):
        #    tch = t[0:gap]
        #    tch = t[interr[j]:interr[j + 1]]
        # Divide the time series in chuncks, according to the data gaps
        if bins == []:
            bins = np.arange(deltat, lentchunk/3., deltat/2.)
        pooled_var = np.zeros(len(bins))
        binsmem = []

        for i in np.arange(len(bins)):
            y_slice = np.array_split(ychunk, int(lentchunk/bins[i]))
            #if int(lentchunk/bins[i])
            pops = []
            varis = []
            if np.shape(y_slice)[0] == 1:
                nelem = int(bins[i]/deltat)
                if nelem < 3:
                    continue
                elif nelem not in binsmem:
                    pops.append(nelem - 1)
                    # Pooled variance or pooled squared MAD
                    if not mad:
                        varis.append(np.var(ychunk[:nelem], ddof=1))
                    else:
                        varis.append(stats.median_absolute_deviation(ychunk[:nelem])**2)
                    pops = np.array(pops)
                    varis = np.array(varis)
                    binsmem.append(np.shape(y_slice)[0])
                    if not RMS:
                        pooled_var[i] = np.sum(pops*varis)/np.sum(pops)
                        print(bins[i]*24*60, pops, pooled_var[i])
                    else:
                        pooled_var[i] = varis**0.5
                    binsmem.append(nelem)
                else:
                    pooled_var[i] = 0.
                    continue
            else:
                if np.shape(y_slice)[0] in binsmem:
                    pooled_var[i] = 0.
                    continue
                else:
                    for j in np.arange(1, np.shape(y_slice)[0]):
                        if len(y_slice[j]) < 3:
                            pooled_var[i] = 0.
                            continue
                        else:
                            pops.append(len(y_slice[j]) - 1)
                            if not mad:
                                varis.append(np.var(y_slice[j], ddof=1))
                            else:
                                varis.append(stats.median_absolute_deviation(y_slice[j])**2)
                    pops = np.array(pops)
                    varis = np.array(varis)
                    binsmem.append(np.shape(y_slice)[0])
                    pooled_var[i] = np.sum(pops*varis)/np.sum(pops)
                            #+ 1) - np.shape(y_slice)[0])
                    #print(bins[i]*24*60, pops, pooled_var[i])
        flag = pooled_var == 0.
        bins = bins[~flag]
        pooled_var = pooled_var[~flag]
        # Check
        if (pooled_var == 0.).any():
            print('Pooled variance: something went wrong.')
            set_trace()

        if plots:
            plt.figure()
            #plt.plot(bins/self.system_pars['Pplanet'], pooled_var, 'o')
            plt.plot(bins, pooled_var, 'o')
            plt.xlabel('Bin width [d]', fontsize=14)
            if not RMS:
                plt.ylabel('Pooled variance', fontsize=14)
            else:
                plt.ylabel('RMS', fontsize=14)
            plt.show()
            set_trace()

        return bins, pooled_var

    def pooled_variance_folded(self, t, y, multfact=1):
        '''
        Calculates the pooled variance (NOT standard deviation)
        of a series of observations - for phase folded LCs.

        Input
        -----
        yarr: arrays containing as many entries as the population samples.
        multfact: needed for variance_P
        See https://en.wikipedia.org/wiki/Pooled_variance
        '''

        pooled_var = np.zeros(multfact)
        # Total duration of LC
        #ttot = (t.max() - t.min())*multfact
        # Time span of every bin
        #bins = bins*self.system_pars['Pplanet']
        y_slice = np.array_split(y, multfact)
        t_slice = np.array_split(t, multfact)
        # for each LC binning
        pops = []
        varis = []
        for i in np.arange(multfact):
            pops.append(len(y_slice[i]) - 1)
            varis.append(np.var(y_slice[i], ddof=1))
        pops = np.array(pops)
        varis = np.array(varis)
        pooled_var = np.sum(pops*varis)/np.sum(pops)
        # Check
        if np.any(varis == 0.):
            print('Pooled variance: something went wrong.')
            set_trace()

        #set_trace()
        return pooled_var

    def scatter_variation(self, sizewin=(50, 100, 150), frange=(1, 1000)):
        '''
        Calculate a running standard deviation of the LC uncertainties,
        using different window widths.

        Input
        -----
        sizewin (tuple): window size to calculate the running std
        france (tuple): interval for the times (frequencies) to plot in the
                        periodogram (and where to look for peaks). In min
        '''

        t, y, yerr = self.prepare_lc(norm_flux=0, clip=1, isplot=0)
        t, y, yerr = self.remove_transits(t, y, yerr, isplot=0)
        yerrd = pd.Series(yerr)

        plt.close('all')
        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)

        for j in sizewin:
            movingstd = yerrd.rolling(j).std()
            good = ~np.isnan(movingstd)
            movingstd = movingstd[good]
            movingt = t[good]
            tf, pgram = phot.periodogram(movingt, movingstd)
            tfmin = tf.to('min')
            # Select interesting frequencies
            ifreq = np.logical_and(tfmin >= frange[0]*u.min, \
                            tfmin <= frange[1]*u.min)
            tfi = tfmin[ifreq]
            pgrami = pgram[ifreq]
            ax1.plot(movingt, movingstd, label='Width = ' + str(j))
            ax2.semilogx(tfi, pgrami, label='Width = ' + str(j))
        ax1.set_xlabel('Time [MJD]', fontsize=16)
        ax1.set_ylabel('LC uncertainty variation', fontsize=16)
        ax2.set_xlabel('Time [min]', fontsize=16)
        ax2.set_ylabel('LSP relative power', fontsize=16)
        ax1.legend(frameon=False, fontsize=16, loc='upper left')
        ax2.legend(frameon=False, fontsize=16, loc='upper left')
        plt.show()
        set_trace()

        return

    def normalize_transit(self, tini, tend, tc, tduration, order, plots=False):
        '''
        Normalize flux centering around an individual transit.
        '''
        flag = np.logical_and(self.t > tini, self.t < tend)
        self.y = self.y[flag]
        self.t = self.t[flag]
        flag2 = np.logical_or(self.t < tc - tduration, self.t > tc + tduration)
        tcorr = self.t[flag2]
        fluxcorr = self.y[flag2]
        fit = np.polyfit(tcorr, fluxcorr, order)
        fiteval = np.polyval(fit, t)

        if plots:
            plt.close('all')
            plt.plot(t, fiteval, 'r-')
            plt.plot(t, flux, '.')
            plt.show()

        return t, flux/fiteval

    def skewness(self, window, plots=False, return_residuals=False):
        '''
        Measure skewness and standard deviations of residuals from LC
        smoothed version.
        '''
        smoothed = smooth.smooth(self.y, window='hanning', \
                                                window_len=window)[:-1]
        residuals = (self.y - smoothed)
        skewness = stats.skew(residuals)
        significance = stats.skewtest(residuals)
        stddev = np.std(residuals)

        if plots:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
            axs[0].plot(self.t, self.y, 'k.', alpha=0.5)
            axs[0].plot(self.t, smoothed, 'orange')
            axs[1].hist(residuals, label='Skewness = ' + str(skewness), \
                    bins=int(len(self.t)/50))
            axs[1].legend()
            plt.show()

        if return_residuals:
            return skewness, stddev, significance.pvalue, residuals
        else:
            return skewness, stddev, significance.pvalue

    def get_PSD(self, autonu=False, numin=10, numax=1e4, nu=0., \
                            logsampling=False, plots=False, verbose=False):
        '''
        First compute a LSP, then scale it to get a PSD

        autonu: let astroy define the frequency array.
        logsampling (bool): linear or log sampling for the frequency array.
        '''

        # Make sure t is a quantity
        if not hasattr(self.t, 'unit'):
            self.t *= u.day
            print('Assuming the time array is in days.')

        ls = timeseries.LombScargle(self.t.to(u.s), self.y, \
                    normalization='psd')
        if autonu:
            freq, periodo = ls.autopower(samples_per_peak=5, nyquist_factor=1)
        elif not autonu and not np.allclose(nu, 0.):
            freq = np.copy(nu)
            periodo = ls.power(freq.to(u.Hz))
        elif not autonu and logsampling:
            freq = np.logspace(np.log10(numin), np.log10(numax), num=1000)*muHz
            periodo = ls.power(freq.to(u.Hz))
        else:
            freq = np.linspace(numin, numax, 2*len(self.t))*muHz
            periodo = ls.power(freq.to(u.Hz))

        periodo[np.isnan(periodo)] = periodo[~np.isnan(periodo)].min()

        # Get FAP - you need a specific normalization for the periodogram
        ls_standard = timeseries.LombScargle(self.t.to(u.s), self.y, \
                    normalization='standard')
        periodo_standard = ls_standard.power(freq.to(u.Hz))
        periodo_standard[np.isnan(periodo_standard)] \
                = periodo_standard[~np.isnan(periodo_standard)].min()

        probabilities = [0.1, 0.05, 0.01]
        try:
            FAP = ls_standard.false_alarm_level(probabilities)
        except FloatingPointError:
            FAP = np.zeros(len(probabilities)) - np.inf

        # Normalization à la Kallinger+2014
        zeta = np.var(self.y)/np.trapz(periodo, x=freq.to(muHz))
        PSD = periodo*zeta

        nu_nyq = 0.5/abs(np.diff(self.t.to(u.s)).min()).value*1e6*muHz

        self.freq = freq
        self.periodo = periodo
        self.periodo_standard = periodo_standard
        self.PSD = PSD.value
        self.zeta = zeta
        self.nu_nyq = nu_nyq
        self.FAP = {}
        for i, f in enumerate(FAP):
            self.FAP[probabilities[i]] = f

        # Parseval's theorem
        if verbose:
            print("\nParseval's theorem checks:")
            print('N*Var (y):', np.var(self.y)*len(self.y), 'ppm^2')
            print('Std (y):', np.std(self.y), 'ppm')
            print('Sum y^2:', np.sum((self.y - np.median(self.y))**2), 'ppm^2')
            print('2*Sum FT^2/N:', 2.*np.sum(periodo), 'ppm^2')
            print('Zeta', zeta)
            print('Max freq:', freq.to(muHz).max(), 'nu_nyq', nu_nyq)

        if plots:
            plt.figure()
            plt.loglog(freq.to(muHz), PSD)
            plt.xlabel('Frequency [$\mu$Hz]', fontsize=14)
            plt.ylabel('PSD [ppm$^2 \, \mu$Hz$^{-1}$]', fontsize=14)
            plt.savefig(self.lc_file + '_PSD.png')
            plt.close()

        return freq, PSD.value

    def fit_PSD(self, profile='kallinger1', plot_fit=True):
        '''
        Perform fit with Harvey or Kallinger+14 model.
        '''

        if not hasattr(self, 'PSD'):
            print('You must first call get_PSD.')
            return

        nu_nyquist = 0.5/abs(np.diff(self.t.to(u.s).value)).min()*u.Hz
        if profile == 'kallinger1':
            params = Parameters()
            params.add('a', value=11., vary=True, min=0.)
            params.add('a1', expr='a*2**0.5')
            params.add('a2', value=0., vary=False)#expr='a1')
            params.add('b1', value=3000., vary=True, min=200., max=7500.)
            params.add('b2_b1', value=1., min=0., max=10000., vary=False)
            params.add('b2', expr='b1 + (b2_b1)')
            params.add('c', value=4., vary=False)
            params.add('Pn', value=0.0340, vary=True, min=0.)
            result = minimize(harvey.kallinger_residuals, params, \
               args=(self.freq.to(muHz).value, self.PSD, 1., False), \
               method='emcee', is_weighted=False, \
               nwalkers=64, burn=1000, steps=2000)
        if profile == 'kallinger2':
            params = Parameters()
            params.add('a1', value=11., vary=True, min=0.)
            params.add('a2', expr='a1')
            params.add('b1', value=200., vary=True, min=0.)
            params.add('b2_b1', value=1., min=0., max=10000., vary=True)
            params.add('b2', expr='b1 + (b2_b1)')
            params.add('c', value=4., vary=False)
            params.add('Pn', value=0.0340, vary=True, min=0.)
            result = minimize(harvey.kallinger_residuals, params, \
               args=(self.freq.to(muHz).value, self.PSD, 1., False), \
               method='emcee', is_weighted=False, \
               nwalkers=64, burn=1000, steps=2000)
        elif profile == 'harvey':
            params = Parameters()
            params.add('A', value=11.523, vary=True, min=0.)
            params.add('b', value=3000., vary=True, min=200., max=7500.)
            params.add('c', value=2., vary=True, min=1., max=10.)
            params.add('Pn', value=0.0340, vary=True, min=0.)
            result = minimize(harvey.harvey_residuals, params, \
                args=(self.freq.to(muHz).value, self.PSD), method='emcee', \
                is_weighted=False, nwalkers=64, burn=2000, steps=4000)

        # Get Max Likelihood Estimates
        highest_prob = np.argmax(result.lnprob)
        hp_loc = np.unravel_index(highest_prob, result.lnprob.shape)
        mle_soln = result.chain[hp_loc]

        emcee_plot = corner.corner(result.flatchain, \
            labels=result.var_names, truths=list(mle_soln))
        plt.savefig(self.lc_file + '_PSD_fit_' + profile + '_corner.png')
        plt.close()

        # Model the MLE: get only those pars which are not fixed
        pvary = []
        for par in result.params:
            if result.params[par].vary:
                pvary.append(par)
        for i, par in enumerate(pvary):
            result.params[par].value = mle_soln[i]
        if profile == 'kallinger1' or profile == 'kallinger2':
            model = harvey.kallinger(result.params, self.freq.to(muHz).value, \
                    verbose=False)
        elif profile == 'harvey':
            model = harvey.harveyfunc_kallinger(result.params, \
                    self.freq.to(muHz).value)

        print('\n', profile, 'profile fit')
        print('----------------------------')
        print('\n', fit_report(result))

        if plot_fit:
            plt.figure()
            plt.loglog(self.freq.to(muHz), self.PSD)
            plt.loglog(self.freq.to(muHz), model, label='Model fit to data PSD')
            plt.legend()
            plt.xlabel('Frequency [$\mu$Hz]', fontsize=14)
            plt.ylabel('PSD [ppm$^2 \, \mu$Hz$^{-1}$]', fontsize=14)
            plt.savefig(self.lc_file + '_PSDfit.png')
            plt.close()

        self.PSD_fit = result
        self.PSD_MLE_model = model
        self.chains = result.flatchain
        fout = open(self.lc_file + '_PSD_fit_' + profile + '.pic', 'wb')
        pickle.dump(self, fout)
        fout.close()

        return result, model

    def prewhitening(self, npeaks=5, nu_range=[0., 500.], plots=False):
        '''
        Remove peaks until there's nothing above 4 sigma.
        Work with periodograms
        '''

        if not hasattr(self, 'periodo'):
            print('You must first call get_PSD.')
            return

        residuals = np.copy(self.y)
        freq2 = np.copy(self.freq)
        periodo2 = np.copy(self.periodo)
        PSD2 = np.copy(self.PSD)

        snr = 100
        iter = 0

        print('Frequency [muHz]\tSNR\tRMS [ppm^2]')
        print('----------------\t---\t-----------')
        while snr > 4 and iter < npeaks:
            flag = np.logical_and(freq2.to(muHz).value > nu_range[0], \
                            freq2.to(muHz).value < nu_range[1])
            freq0 = freq2[flag][periodo2[flag].argmax()]

            # Compute sinudois with best-fit frequency and subtract
            ls = timeseries.LombScargle(self.t, residuals)
            xn = ls.model(self.t, freq0)
            residuals -= xn.value
            freq2, periodo2, _ = phot_analysis.periodogram_FAP(self.t, \
                            residuals, freq=self.freq)
            noiselev = np.std(periodo2[~flag])
            P_freq0 = periodo2[flag].max()
            snr = P_freq0 / noiselev
            print(np.round(freq0.to(muHz).value, 2), '\t', np.round(snr, 2), \
                                '\t', np.round(np.std(periodo2[flag]), 2))

        self.periodo_prewhitened = periodo2

        if plots:
            fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
            axs[0].plot(self.freq, self.periodo, label='Raw')
            axs[0].plot(self.freq, self.periodo_prewhitened, label='Prewhitened')
            axs[1].loglog(self.freq, self.periodo, label='Raw')
            axs[1].loglog(self.freq, self.periodo_prewhitened, label='Prewhitened')
            axs[0].legend()
            axs[1].legend()
            axs[0].set_xlabel('Frequency [$\mu$Hz]', fontsize=14)
            axs[1].set_xlabel('Frequency [$\mu$Hz]', fontsize=14)
            axs[0].set_ylabel('Power [ppm$^2$]', fontsize=14)
            plt.savefig(self.lc_file + '_PSD_prewhitening.png')
            plt.close()

        return self

    def prewhitening_harmonics(self, plots=False, harmonics=5, \
            first_harmonic=[], plot_folded=False):
        '''
        Remove maximum peak and harmonics until there's nothing above 4 sigma.
        Work with periodograms

        Parameters
        ----------
        First harmomic [list in muHz]: if given, this is used instead of the
        highest peak of the periodogram. If multiple elements are given, all
        are used.
        If no value is given, the frequency corresponding to the maximum peak
        is removed.
        '''

        if not hasattr(self, 'periodo'):
            print('You must first call get_PSD.')
            return

        residuals = self.y - np.median(self.y)
        freq2 = np.copy(self.freq)
        periodo2 = np.copy(self.periodo)
        PSD2 = np.copy(self.PSD)

        ls_standard = timeseries.LombScargle(self.t.to(u.s), self.y, \
                                    nterms=harmonics, normalization='standard')
        self.periodo_standard = ls_standard.power(freq2.to(u.Hz))

        #ls = timeseries.LombScargle(self.t, residuals, normalization='psd')
        if first_harmonic == []:
            freq0 = [freq2[np.argmax(periodo2)]]
            max_freq = freq0
        else:
            freq0 = first_harmonic
            max_freq = freq2[freq2.value < 1000][ \
                        np.argmax(periodo2[freq2.value < 1000])]

        residual_snr = []
        for i, freq in enumerate(freq0):
            residual_snr.append([])
            print('Removing harmonics of', freq)
            ls = timeseries.LombScargle(self.t.to(u.s), residuals, \
                                        nterms=harmonics, normalization='psd')
            xn = ls.model(self.t.to(u.s), freq.to(u.Hz))

            residuals -= xn.value
            ls2 = timeseries.LombScargle(self.t.to(u.s), residuals, \
                        normalization='psd')
            freq_ = np.copy(freq2)
            periodo_ = ls2.power(freq_.to(u.Hz))

            noiselev = np.std(periodo_[freq_.value > 6000.])
            # Residual SNR for every harmonic
            for j in np.arange(1, harmonics + 1):
                P_nu0 = periodo_[abs(freq_ - freq*j).argmin()]
                snr = P_nu0 / noiselev
                residual_snr[i].append(snr.value)

        self.periodo_prewhitened = periodo_
        self.y_prewhitened = residuals
        # To compute FAP, another normalization is needed
        ls3 = timeseries.LombScargle(self.t.to(u.s), residuals, \
                    normalization='standard')
        self.periodo_prewhitened_standard = ls3.power(freq_.to(u.Hz))
        probabilities = [0.1, 0.05, 0.01]
        FAP = ls3.false_alarm_level(probabilities)
        self.FAP = {}
        for i, f in enumerate(FAP):
            self.FAP[probabilities[i]] = f

        if plots:
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 10))
            axs[0][0].plot(self.freq, self.periodo_standard, label='Raw')
            axs[0][0].plot(self.freq, self.periodo_prewhitened_standard, \
                            label='Prewhitened')
            axs[0][1].loglog(self.freq, self.periodo_standard, label='Raw')
            axs[0][1].loglog(self.freq, self.periodo_prewhitened_standard, \
                            label='Prewhitened')
            for i, fap in enumerate(self.FAP.keys()):
                axs[0][0].plot([self.freq.to(muHz).value.min(), \
                    self.freq.to(muHz).value.max()], \
                    [self.FAP[fap], self.FAP[fap]], '--', \
                    label=str(int(fap*100)) + '% FAP', c='k', alpha=0.3*(i + 1))
                axs[0][1].loglog([self.freq.to(muHz).value.min(), \
                    self.freq.to(muHz).value.max()], \
                    [self.FAP[fap], self.FAP[fap]], '--', \
                    label=str(int(fap*100)) + '% FAP', c='k', alpha=0.3*(i + 1))
            # Plot significance of remaining peaks after filtering
            fa = [x.to(muHz).value for x in freq0]
            for i, rfreq in enumerate(fa):
                xax = np.arange(harmonics) + 1
                axs[1][0].plot(xax, residual_snr[i], 'o--', \
                            label=str(int(np.round(rfreq))))
            for m in np.arange(1, 6):
                axs[0][1].plot([max_freq.to(muHz).value*m, \
                        max_freq.to(muHz).value*m], \
                        [self.periodo_standard.min(), \
                        self.periodo_standard.max()], 'k--')
                axs[0][1].loglog([max_freq.to(muHz).value*m, \
                        max_freq.to(muHz).value*m], \
                        [self.periodo_standard.min(), \
                        self.periodo_standard.max()], 'k--')
            axs[0][1].legend()
            axs[1][0].legend(title=r'Removed frequency [$\mu$Hz]')
            axs[1][0].set_xlabel(r'Harmonic', fontsize=14)
            axs[1][0].set_ylabel('Residual peak SNR', fontsize=14)
            # Plot time series folded to the mean CHEOPS frequency
            fa = [x.to(u.Hz).value for x in freq0]
            fold_per = (1./np.mean(fa)*u.s).to(u.day)
            phase = (self.t.to(u.day) % fold_per)/fold_per
            axs[1][1].plot(phase, self.y - np.median(self.y), '.', label='Raw')
            axs[1][1].plot(phase, residuals, '.', label='Prewhitened')
            axs[1][1].legend(loc='upper right')
            axs[0][0].legend(loc='upper right')
            for freq in first_harmonic:
                col = (np.random.random(), np.random.random(), np.random.random())
                for j in np.arange(1, harmonics + 1):
                    freq_i = freq.to(muHz).value*j
                    axs[0][0].plot([freq_i, freq_i], \
                        [self.periodo_prewhitened_standard[self.freq.value < 3000.].min()*0.5, \
                        self.periodo_standard[self.freq.value < 3000.].max()*1.1], '--', c=col)
                    axs[0][1].loglog([freq_i, freq_i], \
                        [self.periodo_prewhitened_standard[self.freq.value < 3000.].min()/10., \
                        self.periodo_standard[self.freq.value < 3000.].max()*10], '--', c=col)
            axs[0][0].set_ylim(self.periodo_prewhitened_standard[self.freq.value < 3000.].min()*0.5, \
                    self.periodo_standard[self.freq.value < 3000.].max()*1.1)
            axs[0][1].set_ylim(max([self.periodo_prewhitened_standard[self.freq.value < 3000.].min()/10., \
                    1e-6]), \
                    self.periodo_standard[self.freq.value < 3000.].max()*10)
            axs[0][0].set_xlim(0., 3000.)
            axs[0][0].set_xlabel('Frequency [$\mu$Hz]', fontsize=14)
            axs[0][1].set_xlabel('Frequency [$\mu$Hz]', fontsize=14)
            axs[0][0].set_ylabel('LS Power [ppm$^2$]', fontsize=14)
            axs[0][1].set_ylabel('LS Power [ppm$^2$]', fontsize=14)
            axs[1][1].set_xlabel('Roll-angle phase', fontsize=14)
            axs[1][1].set_ylabel('Relative flux [ppm]', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.lc_file + '_periodo_prewhitened.png')
            plt.close()

        return self

    def correlated_noise(self, maxbin, interval=50, plots=False):
        '''
        Compute correlated noise level for a given time bin, following
        Pont+2006.

        t and maxbin must in the same units.
        '''

        print('Computing red noise curve...')

        # Compute correlated noise level
        resolution = abs(np.diff(self.t)).min()
        l_win = int(maxbin/resolution)
        V_n, white = [], []
        nbins =  np.arange(1, l_win + 1, interval)
        for n in nbins:
            F_j = signal.medfilt(self.y, kernel_size=n)
            V_n.append(np.std(F_j))
            # White noise
            try:
                white.append(np.std(self.y)/np.sqrt(n))
            except FloatingPointError:
                white.append(np.sqrt(np.sum((self.y - np.mean(self.y))**2) \
                    /len(self.y))/np.sqrt(n))

        if plots:
            plt.figure()
            plt.plot(nbins, np.array(V_n)*1e6, label='red')
            plt.plot(nbins, np.array(white)*1e6, label='white')
            plt.legend()
            plt.xlabel('Data points')
            plt.ylabel('Noise level [ppm]')
            plt.show()
            set_trace()

        return np.array(nbins), np.array(V_n), np.array(white)

    def rebin_old(self, binning_factor):

        self.t = self.t[::binning_factor]
        #ytot = signal.medfilt(ytot, kernel_size=7)[::7]

        yerrtot_binned = np.zeros(len(self.t))
        for i in np.arange(len(yerrtot_binned)):
            bin_i = i*binning_factor
            bin_fin = i*binning_factor + binning_factor
            yerrtot_binned[i] += np.std(self.y[bin_i : bin_fin])
        self.yerr = np.zeros(len(self.t)) \
                + np.sqrt(np.median(self.yerr)**2/binning_factor \
                + yerrtot_binned**2)

        ytot_binned = np.zeros(len(self.t))
        for i in np.arange(len(ytot_binned)):
            bin_i = i*binning_factor
            bin_fin = i*binning_factor + binning_factor
            ytot_binned[i] += np.mean(self.y[bin_i : bin_fin])
        self.y = np.copy(ytot_binned)

        return self.t, self.y, self.yerr

    def rebin(self, binning_factor):

        tnew = self.t[::binning_factor]
        #ytot = signal.medfilt(ytot, kernel_size=7)[::7]

        yerrtot_binned = np.zeros(len(tnew))
        ytot_binned = np.zeros(len(tnew))
        for i in np.arange(len(yerrtot_binned)):
            bin_i = i*binning_factor
            bin_fin = i*binning_factor + binning_factor
            weights = self.yerr[bin_i : bin_fin]**-2
            yerrtot_binned[i] += np.sqrt(1./np.sum(weights))
            ytot_binned[i] += np.average(self.y[bin_i : bin_fin], \
                        weights=weights)

        self.t = np.copy(tnew)
        self.y = np.copy(ytot_binned)
        self.yerr = np.copy(yerrtot_binned)
        
        return self.t, self.y, self.yerr
