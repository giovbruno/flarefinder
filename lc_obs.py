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
