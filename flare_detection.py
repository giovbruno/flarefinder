import os
home = os.path.expanduser('~')
import sys
import glob
from scipy.signal import medfilt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import optimize, stats
import gp_utilities
from astropy.timeseries import LombScargle
import astropy.units as u
muHz = u.def_unit('muHz', 1e-6*u.Hz)
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy import timeseries
import celerite2
from celerite2 import terms
import numpy as np
import copy
import lc_obs
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import peakutils.peak as peakut
import smooth
from lmfit import Parameters, fit_report, minimize
#from pyts.decomposition import SingularSpectrumAnalysis as SSA
from pdb import set_trace
import warnings
warnings.filterwarnings("ignore")
import flare_class
import pickle
import models
import rebin

def find_flares(lcfile, stdflare=0., \
        tstar=5777., rstar=1., wth=None, fth=None, peaki=[], \
        smoothmode='smooth', plot_flat=False, rednoise=[], \
        flarepeakfilter=10, flare_threshold=5., outlier_thresh=8., \
        fit_continuum=0, force_fit=False, clip=True, normalise=True, \
        complexity=5, saveplots='', min_datapoints=3, rebin=0,
        filt_kernel_size=11):
    '''
    Parameters
    ----------
    pxdivide for 25 s LCs: 1000. This factor is multiplied for larger cadences.
    peaki: list of peaks can be provided

    rednoise (list of 3 elements): it can be provided here, or is calculated
                                   during smoothing (if smoothmode is not None)
    filt_kernel_size: kernel size for the median filter to be used in the flare
              edges definition

    Data needs to be normalised to 1.
    '''

    if type(lcfile) == list:
        t, y = lcfile[0], lcfile[1]
        if np.shape(lcfile)[0] == 3:
            yerr = lcfile[2]
        else:
            yerr = None
    elif type(lcfile) == str and \
                (lcfile.endswith('.dat') or lcfile.endswith('.txt')):
        print('\nLC file:', lcfile.split(lcfolder)[1], '\n')
        t, y = np.loadtxt(lcfile, unpack=True)
        yerr = None
    elif type(lcfile) == str and lcfile.endswith('.fits'):
        lc = fits.open(lcfile)
        try:
            t = lc[1].data['TIME']
            y = lc[1].data['PDCSAP_FLUX']
            yerr = lc[1].data['PDCSAP_FLUX_ERR']
        except TypeError: # Problem with the dataset
            return []

    # Get or determine uncertainties
    if yerr is None:
        yerr = np.zeros(len(t)) + stdflare
        #filt1 = smooth.smooth(y, window_len=filter_std)
        #yerr = np.zeros(len(t)) + np.std(y - filt1)

    # Remove all NaNs
    flag = np.isnan(y)
    t, y, yerr = t[~flag], y[~flag], yerr[~flag]

    if rebin > 1:
        light_curve = lc_obs.LC(t=t, y=y, yerr=yerr)
        t, y, yerr = light_curve.rebin(rebin)

    # Normalise LC
    if normalise:
        yerr /= abs(np.median(y))
        y /= abs(np.median(y))

    # Remove stellar variability - this is to fit the flare profiles
    if smoothmode is not None:
        xf, yf, yferr, ls_quiet, rednoise \
            = smooth_LC(t, y, yerr, plots=False, mode=smoothmode, \
                savefile=saveplots.replace('.pdf','_LC_detrended.pic').replace( \
                '.fits', '_LC_detrended.pic'))
        # These will be the points for the smoothing calculation
        filt_var_int = interp1d(xf, yf, bounds_error=False, \
                    fill_value='extrapolate')
        filt_var = filt_var_int(t)
        if len(filt_var) == len(y) + 1:
            filt_var = filt_var[:-1]
    else:
        xf, yf, yferr = np.copy(t), np.zeros(len(t)), np.zeros(len(t)) \
                        + np.median(yerr)
        #y /= np.median(y)
        freq, power = [], []
        ls_quiet = np.zeros(len(y))
        filt_var = np.copy(yf)

    yflat = y - filt_var

    # Remove outliers
    tini = np.copy(t)
    if clip:
        #plt.close('all')
        #plt.plot(t, yflat)
        yflatfilt = medfilt(yflat, kernel_size=3)
        mask = sigma_clip(yflat - yflatfilt, sigma=outlier_thresh)
        t = t[~mask.mask]
        yflat = yflat[~mask.mask]
        #yerr = yerr[~mask.mask]
        tini = tini[~mask.mask]
        filt_var = filt_var[~mask.mask]
        y = y[~mask.mask]
        #noiselev = noiselev[~mask.mask]
        #plt.plot(t, yflat)
        #plt.show()
        #set_trace()
    # Both on LC and on binned version of LC
    if stdflare == 0.:
        #stdflare = np.median(yerr)#nanstd(yfilt)
        stdflare = np.copy(yferr)

    # This is to find the flare peaks
    #filt_peaks = smooth.smooth(yflat, window_len=flarepeakfilter)[:len(y)]
    #yfilt = yflat - filt_peaks
    #noiselev = yerr#np.sqrt(yerr**2. + stdflare**2.)
    noiselev = np.median(yferr)
    if noiselev == 0:
        set_trace()
    #yerr = np.copy(noiselev)
    if peaki == []:
        peaki = list(peakut.indexes(yflat, thres=flare_threshold*noiselev, \
                        min_dist=10, thres_abs=True))
        #peaki.append(peakut.indexes(medfilt(y, kernel_size=3), \
        #        thres=flare_threshold*stdflare, min_dist=10, thres_abs=True))
        #peaki = np.sort(np.hstack(peaki))

    print('Detected peaks:', len(peaki))

    #if clip:
    #    # Elements to remove that are not among peaks
    #    mask2 = np.setdiff1d(np.where(mask.mask), peaki)
    #    t = np.delete(t, mask2)
    #    yflat = np.delete(yflat, mask2)
    #    yerr = np.delete(yerr, mask2)
    #    tini = np.copy(t)
    #    filt_var = np.delete(filt_var, mask2)
    #    y = np.delete(y, mask2)
    #    noiselev = np.delete(noiselev, mask2)
    #    # And look for peaks again
    #    peaki = list(peakut.indexes(yflat, thres=flare_threshold*noiselev, \
    #                    min_dist=10, thres_abs=True))

    # Plot flattended LC
    if plot_flat:
        plt.figure(figsize=(12, 5))
        plt.plot(tini, y, '.', alpha=0.5, label='Raw flux')
        plt.plot(tini, filt_var, label='Quiet flux model')
        plt.plot(tini, filt_var + flare_threshold*noiselev, '--', \
                    label='Detection threshold')
        #plt.plot(t, medfilt(y, kernel_size=3), '--', label='Binned LC')
        for pp in peaki:
            plt.plot([t[pp], t[pp]], [y[pp] + 5*yerr.max(), \
                    y[pp] + 10*yerr.max()], 'c', linewidth=2)
        plt.legend()
        plt.xlabel('Time [days]', fontsize=14)
        plt.ylabel('Relative flux', fontsize=14)
        plt.savefig(saveplots.replace('.pdf','_LC.pdf').replace( '.fits', '_LC.pdf'))
        plt.show()
        set_trace()
        plt.close('all')

    flarespar = fitflares(t, yflat, yerr, peaki, noiselev, traw=np.copy(t), yraw=y, \
            threshold=flare_threshold, rednoise=rednoise, \
            tstar=tstar, rstar=rstar, wth=wth, fth=fth, \
            fit_continuum=fit_continuum, force_fit=force_fit, \
            complexity=complexity, min_datapoints=min_datapoints, \
            plotname=saveplots.replace('.pdf','_flares.pdf').replace('.fits', \
                    '_flares.pdf'), filt_kernel_size=filt_kernel_size)

    plt.close('all')

    return flarespar

def fitflares(t, yflat, yerr, peaki, noise_level, threshold=4., complexity=6, \
            tstar=None, rstar=None, wth=None, fth=None, rednoise=None, \
            force_fit=False, fit_continuum=0, plotname='', check_gaps=True, \
            traw=None, yraw=None, min_datapoints=3, filt_kernel_size=11):
    '''
    Iterative flare fit routine. No need for two filtered versions of the LC.
    Build a light curve model with all fitted flares.

    Parameters
    ----------
    time and fluxes for smoothed light curve.
    peaki (list): flare peaks
    force_fit: assume a flare candidate is a flare (avoid checks)
    explore_all_peaks: try fit until max complexity, avoiding stopping criteria
    min_datapoints: minimum flare duration in data points
    filt_kernel_size: kernel size for the median filter to be used in the flare
              edges definition

    Returns
    -------
    flarespar: dict with fitted flare parameters
    chi2min: min reduced chi2 value, to avoid overfitting

    Candidate = 0: non-validated flare
    Candidate = 1: validated flare
    '''

    # This is used to determine flare length, by removing some noise
    filtflare = medfilt(yflat, kernel_size=filt_kernel_size)

    flarespar = []
    # This is to check if this flare was already fitted on a multi-peak one
    pplot = PdfPages(plotname)

    nflares = 0
    tuntil = 0
    trawcopy = np.copy(traw)
    yrawcopy = np.copy(yraw)
    for i, peak in enumerate(peaki):
        print('Flare #', i + 1, '/', len(peaki))

        # Get previous fitted peaks
        if peak <= tuntil:
            print('Already fitted')
            continue

        if len(t) - peak < 10 or peak < 10:
            print('Too close to dataset edge')
            continue

        noiselev = np.median(noise_level)

        j = -1
        flagleft = t < t[peak]
        while abs(j) < len(filtflare[flagleft]) - 1 \
                            and filtflare[flagleft][j] > noiselev:
            j -= 1
        tini = abs(t \
            - t[flagleft][abs(t[flagleft] - t[flagleft][j]).argmin()]).argmin()
        # This will be later used for the dip
        tbeg = np.copy(tini)

        j = 1
        flagright = t > t[peak]
        while j < len(filtflare[flagright]) - 1 \
                            and filtflare[flagright][j] > noiselev:
            j += 1
        tend = abs(t - t[flagright][abs(t[flagright] \
                    - t[flagright][j]).argmin()]).argmin()
        tfin = np.copy(tend)

        # Estimate flare duration and compare with correlated noise level
        # (useful when the detrending is not fully effective)
        duration_estimate = tend - tini
        rednoise_level = rednoise[1][ \
                    abs(np.array(rednoise[0]) - duration_estimate).argmin()]
        significance = abs(yflat[peak])/rednoise_level
        if significance < 1.:
            continue

        # Add points to help catch the continuum and dips
        # This happened with CHEOPS data!
        dd = np.diff(t)
        #if (dd == 0.).any():
        #    set_trace()
        maxleft = int(1./24./dd[dd > 0].min())
        maxright = int(1./24./dd[dd > 0].min())
        indexleft = 0
        while tini - indexleft > 0 and indexleft < maxleft:
            indexleft += 1
            if filtflare[tini - indexleft] > noiselev:
                break

        indexright = 0
        while tend + indexright < len(t) - 1 and indexright < maxright:
            indexright += 1
            if filtflare[tend + indexright] > noiselev:
                break

        tini -= indexleft
        tend += indexright
        tuntil = np.copy(tend)
        tw = t[tini:tend] - t[peak]
        traw = trawcopy[tini:tend] - t[peak]
        yw = yflat[tini:tend]
        yraw = yrawcopy[tini:tend]
        yerrw = yerr[tini:tend]
        j = abs(tw).argmin()

        # Flare normalisation
        #tc = np.concatenate((t[tini:tbeg], t[tfin:tend]))
        #yc = np.concatenate((yflat[tini:tbeg], yflat[tfin:tend]))
        #continuum = np.polyfit(tc, yc, 1)
        #plt.close('all')
        #plt.plot(tw, yw)
        #plt.plot(tw, np.polyval(continuum, t[tini:tend]))
        #plt.show()
        #set_trace()

        if len(tw) < 3:
            print('Too short: Probably an outlier')
            plt.close('all')
            #set_trace()
            continue

        #if tini == 0 or tend == len(t) - 1:
        #    print('Probably an edge effect')
        #    plt.close('all')
        #    continue

        # Reject features that are likely outliers or edge effects
        # If the end of the flare is not identified either, this might
        # be some sort of correlated noise
        #yw = np.median(yw)
        #if (yw[j + 1] <= 3.*noiselev + myw or yw[j + 2] <= 3.*noiselev + myw) \
        #                and not force_fit:
        myw = 0.
        #if (yw[j : j + 5] <= 3.*noiselev).any() and not force_fit:
        if (yw[j + 1: j + min_datapoints] <= 2.*noiselev).any():
            #    or (yw[j + 1] < yw[j + 2]) and len(tw) < 10:
            #or (yw[j + 3: j + 5] <= 2.*noiselev).any()) and not force_fit:
            print('Too short: probably an outlier')
            continue

        if (np.diff(tw) > np.median(np.diff(t))*10).any() and check_gaps:
            # Only consider the flux within gaps
            gaps = np.where(np.diff(tw) > np.median(np.diff(t))*10)[0]
            # initialize the closest indices as the first and second elements of the array
            #idx1 = 0
            #idx2 = 1
            if len(gaps) > 1:
                #idx1 = 0
                #idx2 = 1
                # Go look into the first and last LC segment
                if gaps[0] > j:
                    tw = tw[:gaps[0]]
                    yw = yw[:gaps[0]]
                    traw = traw[:gaps[0]]
                    yraw = yraw[:gaps[0]]
                    yerrw = yerrw[:gaps[0]]
                elif j > gaps[-1]:
                    tw = tw[gaps[-1] + 1:]
                    yw = yw[gaps[-1] + 1:]
                    traw = traw[gaps[-1] + 1:]
                    yraw = yraw[gaps[-1] + 1:]
                    yerrw = yerrw[gaps[-1] + 1:]
                else:
                # loop through the array to find the closest indices
                    for i in range(len(gaps) - 1):
                        if gaps[i] <= j <= gaps[i + 1]:
                            #if j - gaps[i] < gaps[i + 1] - j:
                            idx1 = i
                            idx2 = i + 1
                            break
                    tw = tw[gaps[idx1] + 1 : gaps[idx2]]
                    yw = yw[gaps[idx1] + 1 : gaps[idx2]]
                    traw = traw[gaps[idx1] + 1 : gaps[idx2]]
                    yraw = yraw[gaps[idx1] + 1 : gaps[idx2]]
                    yerrw = yerrw[gaps[idx1] + 1 : gaps[idx2]]
            else:
                if gaps[0] > j:
                    idx1 = 0
                    idx2 = gaps[0]
                else:#if j > gaps[0]:
                    idx1 = gaps[0] + 1
                    idx2 = -1
                #print('Idx 1, 2:', idx1, idx2)
                tw = tw[idx1 : idx2]
                yw = yw[idx1 : idx2]
                traw = traw[idx1 : idx2]
                yraw = yraw[idx1 : idx2]
                yerrw = yerrw[idx1 : idx2]
            #set_trace()
        elif not check_gaps:
            tw = tw[2:-2]
            yw = yw[2:-2]
            traw = traw[2:-2]
            yraw = yraw[2:-2]
            yerrw = yerrw[2:-2]

        # Other check here
        # 3 is the number of the Gaussian fit parameters
        if len(tw) < fit_continuum + 3:
            print('Probably an outlier')
            continue

        # This is necessary because of the data gaps
        #traw -= t[peak]
        #traw_ini = abs(traw - tw[0]).argmin()
        #traw_fin = abs(traw - tw[-1]).argmin()
        aflare = flare_class.flare(tdata=tw, ydata=yw, yerrdata=yerrw, \
                tbeg=t[tbeg] - t[peak], tend=t[tfin] - t[peak], \
                Tstar=tstar, Rstar=rstar, tpeak=t[peak], noise_level=noiselev, \
                rednoise=rednoise, traw=traw, yraw=yraw)

        # First, the case with no flare: if this fit has the best AIC, the
        # flare is not validated
        aflare.fit_line()
        #aflare.fit_gaussian(fit_continuum)
        aflare.fit_flare_profile(complexity, threshold, \
                    fit_continuum=fit_continuum)

        if aflare.npeaks > 0:
            nflares += 1

            # Test dip hypothesis
            result_dip, dip_delta_aic, significance = aflare.dip_fit()
            aflare.result_dip = result_dip
            aflare.dip_delta_aic = dip_delta_aic
            aflare.dip_significance = significance
            # Same for CME
            #xmin = (t[tfin] - t[peak])/aflare.t12.value
            #if xmin == tw.max():
            #    xmin -= abs(np.diff(t)[np.diff(t) > 0.]).min()#/aflare.t12.value
            #xmax = tw.max()#/aflare.t12.value
            #if xmin == xmax:
            #    xmin -= abs(np.diff(t)[np.diff(t) > 0.]).min()#/aflare.t12.value
            #result_cme, cme_delta_aic, significance = aflare.dip_fit(xmin, xmax)
            #aflare.result_cme = result_cme
            #aflare.cme_delta_aic = cme_delta_aic
            #aflare.cme_significance = significance

            # To uncomment, make sure plotname is a string
            aflare.plot_models(plot_instance=pplot, plotname=plotname, \
                        plot_LS=False)

            flarespar.append(aflare)

    pplot.close()
    if nflares == 0:
        os.system('rm ' + plotname)

    return flarespar

def smooth_LC(t, f, ferr, plots=False, mode='smooth', compute_rednoise=True, \
            savefile=''):
    '''
    Find a smoothed verision of the LC, removing flares through iterative
    sigma-clipping

    Timewindow: use the number of data points to compute a time window (hours)
    or use it as it is.
    '''

    if mode == 'spline':
        ndatapoints = int(120./(np.median(np.diff(t))*86400.)) # As TESS SC
        x = t[::ndatapoints]
        y = f[::ndatapoints]
    elif mode == 'smooth':
        freq, power = LombScargle(t*u.day, f).autopower( \
                maximum_frequency=0.5/np.diff(t*u.day).min())
        win = freq[power.argmax()]**-1*24./5./u.day # in hours
        smooth_factor = int(win*60*60./(np.median(np.diff(t))*86400.))
    elif mode == 'prewhitening':
        # First reject large outliers, like transits
        fmed = medfilt(f, kernel_size=501)
        fclip = sigma_clip(f - fmed, sigma=3)
        tcl = t[fclip.mask == False]
        ycl = f[fclip.mask == False]
        y = np.copy(f)
        if np.diff(tcl).min() < 100/86400.:
            takevery = 1
        elif np.diff(tcl).min() > 100/86400. and np.diff(tcl).min() < 200/86400.:
            takevery = 3
        y_prewhitened = prewhitening(tcl[::takevery], ycl[::takevery], \
                        plots=False, verbose=False, npeaks=100, nurange=[1, 1000])
        y_model = ycl[::takevery] - y_prewhitened
        #yp_int = interp1d(tcl[::takevery], y_model, bounds_error=False, \
        #                fill_value='extrapolate')
        #y_model = yp_int(t)
    #elif mode == 'SSA':
    #    X = f.reshape(1, -1)
    #    transformer = SSA(window_size=11)
    #    X_new = transformer.transform(X)
    elif mode == 'GP': # Incomplete #
        freq, power = LombScargle(t*u.day, f).autopower( \
                maximum_frequency=0.5/np.diff(t*u.day).min())
        P = 1./freq[power.argmax()].value
        # Define GP
        term1 = terms.RotationTerm(sigma=np.std(f), period=P, Q0=1, dQ=1, f=0.5)
        term2 = terms.SHOTerm(sigma=1e-4, w0=0.1/24., Q=2.**0.5)
        gp = celerite2.GaussianProcess(term1 + term2, mean=np.median(f))
        gp.compute(t, yerr=ferr)
        #initial_params = [np.median(f), np.std(f), P, 0.1, 0.1, 0.5, #np.median(ferr)]
        initial_params = [np.median(f), np.std(f), P, 0.1, 0.1, 0.5, 1e-3, 0.5/24., np.median(ferr)]
        bounds = [(np.median(f) - np.median(ferr)*3., np.median(f) + np.median(ferr)), \
                (1e-6, 3.*np.std(f)), (0.1, 100.), (0.1, 10.), \
                (0.1, 10.), (1e-6, 1. - 1e-6), (1e-5, 1e-2), (0.05/24., 3./24.), (np.median(ferr), 3.*np.median(ferr))]
        soln = optimize.minimize(gp_utilities.neg_log_like, initial_params, \
                    bounds=bounds, method="L-BFGS-B", args=(gp, t, f, ferr))
        opt_gp = gp_utilities.set_params(soln.x, gp, t, ferr)
        y_model = opt_gp.predict(f, t, return_var=False)
        y_model_GP = np.copy(y_model)
        y = np.copy(f)
        x = np.copy(t)
    elif mode == 'pspline':
        flatten_lc, y_model = flatten(t, f, method='pspline', edge_cutoff=0.05, \
                        stdev_cut=3, return_trend=True, break_tolerance=0.1)
        # Remove NaNs from y_model
        nn = np.isnan(y_model)
        y_model = y_model[~nn]
        y = f[~nn]
        x = t[~nn]

    # At most 1% of the points can be removed before smoothing. Start
    # with a 3-sigma rejection threshold, then increased if too many points
    # are rejected
    #if mode != 'prewhitening':# and mode != 'GP':
    f_removed = 10.
    n_removed = 0
    sigma_threshold = 1.
    iter = 0
    if mode != 'pspline':
        while f_removed > 0.03:
            x = np.copy(t)
            y = np.copy(f)
            yerr = np.copy(ferr)
            n_removed = 0
            nmask = 10
            while nmask > 0:
                if mode == 'smooth':
                    y_model = smooth.smooth(y, window_len= \
                            min([int(len(y)/5.), smooth_factor]))[:len(x)]
                elif mode == 'spline':
                    #model = splrep(x, y, k=3, s=0.5)
                    #y_model = splev(x, model)
                    model = UnivariateSpline(x, y, k=3, s=0.016)
                    y_model = model(x)
                elif mode == 'GP':
                    soln = optimize.minimize(gp_utilities.neg_log_like, \
                        initial_params, method="L-BFGS-B", \
                        bounds=bounds, args=(gp, x, y, yerr))
                    opt_gp = gp_utilities.set_params(soln.x, gp, x, yerr)
                    gp.compute(x, yerr=yerr)
                    y_model = opt_gp.predict(y, x, return_var=False)
                elif mode == 'prewhitening':
                    y_prewhitened = prewhitening(x, y, \
                            npeaks=100, nurange=[1, 1000])
                    y_model = y - y_prewhitened
                yc = sigma_clip(y - y_model, sigma=sigma_threshold)
                nmask = np.sum(yc.mask)
                x = x[~yc.mask]
                y = y[~yc.mask]
                yerr = yerr[~yc.mask]
                n_removed += np.sum(nmask)
            f_removed = 1. - (len(t) - n_removed)/len(t)
            sigma_threshold += 1
        #else:
        #    x = np.copy(t)

    if mode == 'GP':
        y_model_GP = opt_gp.predict(y, t, return_var=False)
        # Evaluate uncertainties for a lower number of data points,
        # then interpolate
        #opt_gp = gp_utilities.set_params(soln.x, gp, x[::100], yerr[::100])
        y_model_i, y_model_var_i = opt_gp.predict(y, x[::100], return_var=True)
        #y_model_var_i /= 10.
        var_int = interp1d(x[::100], y_model_var_i, bounds_error=False, \
                fill_value='extrapolate')
        y_model_err = var_int(t)**0.5

    if plots:
        plt.plot(t, f, 'b', alpha=0.2)
        plt.plot(x, y, 'b')
        if mode == 'spline':
            plt.plot(t, model(t), 'orange', linewidth=3)
        elif mode == 'smooth' or mode == 'pspline':
            plt.plot(x, y_model, 'orange', linewidth=3)
        elif mode == 'prewhitening':
            plt.plot(x, y_model, 'orange', linewidth=3)
        elif mode == 'GP':
            plt.plot(t, y_model_GP, 'r', linewidth=3)
            plt.fill_between(t, y1=y_model_GP + y_model_err, \
                        y2=y_model_GP - y_model_err, color='orange')
        plt.xlabel('Time [days]', fontsize=14)
        plt.ylabel('Normalised flux', fontsize=14)
        plt.show()
        set_trace()
        plt.close('all')

    # Get PSD of bit of flare-free LC (not normalized)
    lcminutes = (x - x.min())*u.day.to(u.min)
    xlc = np.logical_and(lcminutes > 0, lcminutes < 200)
    ls = timeseries.LombScargle(lcminutes[xlc]*u.min.to(u.s), y[xlc], \
                normalization='standard', fit_mean=True)
    #freq = np.linspace(167*1e-6, 10000*1e-6, 1000)
    #freq, PSD = ls.autopower()#freq)

    # Evaluate red noise level on the residuals
    if compute_rednoise:
        lightc = lc_obs.LC(t=x, y=(y - y_model))
        bins, red, white = lightc.correlated_noise(3600./86400., \
                interval=10, plots=False)
    else:
        bins, red, white = [0., 0., 0.]

    # Save smoothed LC for later inspection
    if savefile != '':
        fout = open(savefile, 'wb')
        pickle.dump([x, y - y_model], fout)
        fout.close()

    if mode == 'smooth' or mode == 'prewhitening':
        return x, y_model, np.std(y - y_model), ls, [bins, red, white]
    elif mode == 'pspline':
        flatten_clipped = sigma_clip(y - y_model, sigma=3)
        # This avoids a FloatingPointError
        stddev = np.sqrt(np.sum((flatten_clipped \
                - np.mean(flatten_clipped))**2)/(len(flatten_clipped) - 1))
        return x, y_model, stddev, ls, [bins, red, white]
    elif mode == 'spline':
        return t, model(t), np.std(y - model(t)), ls, [bins, red, white]
    elif mode == 'GP':
        return t, y_model_GP, np.std(y - y_model), ls, [bins, red, white]

def prewhitening(t, y, nurange=[1., 100], npeaks=10, plots=False,
    verbose=False):
    '''
    Remove sinusoidal signals from light curve.
    '''

    yc = np.copy(y)
    t2 = (t*u.day).to(u.s).value
    freq, power = LombScargle(t2, y).autopower( \
                minimum_frequency=1e-6, maximum_frequency=10000e-6)#277e-6)
    freq0, power0 = np.copy(freq), np.copy(power)
    flag = np.logical_and(freq*1e6 >= nurange[0], freq*1e6 <= nurange[1])
    freq *= u.Hz
    noiselev = 0.

    for j in np.arange(npeaks):
        nu0 = freq[flag][power[flag].argmax()]
        ls = LombScargle(t2, yc)
        xn = ls.model(t2, nu0.value)
        yc -= xn
        freq, power = LombScargle(t2, yc).autopower( \
                minimum_frequency=1e-6, maximum_frequency=100e-6)
        flag = np.logical_and(freq*1e6 >= nurange[0], freq*1e6 <= nurange[1])
        if noiselev == 0.:
            noiselev = np.std(power[freq*1e6 > 5000])#np.std(power[flag])
        P_nu0 = power[flag].max()
        snr = P_nu0 / noiselev
        #fap = ls.false_alarm_probability(P_nu0)*100.
        if verbose:
            print('Iteration:', j, ' FAP:', fap, '%')
        #if fap > 1:
        #    break
        if snr < 4:
            break
        freq *= u.Hz

    if plots:
        plt.figure()
        plt.plot(t2/60., y - np.median(y), label='Original - median')
        plt.plot(t2/60., yc, label='Prewhitened')
        plt.xlabel('Time [min]', fontsize=14)
        plt.ylabel('Flux [units?]', fontsize=14)
        plt.legend()
        plt.figure()
        plt.semilogx(freq0*1e6, power0, label='Original')
        plt.semilogx(freq*1e6, power, label='Prewhitened')
        plt.xlabel(r'$\mu$Hz', fontsize=14)
        plt.ylabel('LS power', fontsize=14)
        plt.show()
        set_trace()

    return yc

def test_cadence_effect():
    '''
    Plot a flare profile at different cadences, to test their impact.
    '''

    ampl_3s = []
    ampl_21s = []
    ampl_60s = []
    durat_3s = []
    durat_21s = []
    durat_60s = []
    true_ampl = []
    true_durat = []

    for j in np.arange(1000):
        c2 = 29.30
        c1 = 0.55
        par = Parameters()
        par.add('fwhm0', value=stats.loguniform.rvs(0.5, 20))
        logL = c2 + np.random.normal(loc=0., scale=0.06) \
            + (c1 + np.random.normal(loc=0., scale=0.05))*np.log10(par['fwhm0'].value)
        par.add('ampl0', value=10**logL)
        par.add('tpeak0', value=0.0)
        t = np.arange(-120., 6000., 3./60.)
        y = models.flare_model_mendoza(t, par, plots=False)
        y /= y.max()
        true_ampl.append(par['ampl0'].value)
        true_durat.append(np.ptp(t[y/y.max() > 1e-6]))

        plt.plot(t, y, 'k', label='No noise')

        # Get noise level from quiescent luminosity
        zeropoint = 2.49769e-9
        lambdaeff = 5850.88
        distance = 10*u.pc
        mag = 9.#np.random.uniform(7, 12)
        # Flux densities --> fluxes with effective wavelength
        F0 = zeropoint*u.erg/u.cm**2/u.s/u.A
        F = F0*10**(-mag/2.5)*lambdaeff*u.A
        L_quiesc = 4.*np.pi*distance.to(u.cm)**2*F

        noiselevel = 240*1e-6#*L_quiesc.value#stats.loguniform.rvs(1e-4, 1e-2)
        y += np.random.normal(loc=0., scale=noiselevel, size=len(t))
        #if np.sum(flag) > 0
        tini, tfin = find_duration_simple(t, y, noiselevel)
        #aflare = flare_class.flare(tdata=t[tini - 10:tfin + 10]/(24*60), \
        #        ydata=y[tini - 10:tfin + 10]/y.max(), \
        #        noise_level=noiselevel/y.max(), \
        #        yerrdata=np.zeros(tfin-tini + 20)+noiselevel/y.max(), \
        #        tbeg=t[tini]/(24.*60.), tend=t[tfin]/(24.*60.), tpeak=0.)
        #aflare.fit_line()
        #aflare.fit_flare_profile(1, 4)
        #try:
        #    #durat_3s.append(aflare.durations[0].to(u.min).value)
        #    durat_3s.append(aflare.result_nodip.params['fwhm0'].value*24*60.)
        #except AttributeError:
        #    set_trace()
        #ampl_3s.append(aflare.result_nodip.params['ampl0']*y.max())
        #else:
        #    print(par)
        #    set_trace()
        plt.plot(t, y, '+-', label='3 s')
        plt.plot([t[tini], t[tini]], [y.min(), 2*y.max()], '--', c='tab:blue')
        plt.plot([t[tfin], t[tfin]], [y.min(), 2*y.max()], '--', c='tab:blue')

        rebfactor = [7, 20]
        labels = ['21 s', '1 min']
        for i, reb in enumerate(rebfactor):
            ti = rebin.rebin(t, reb)
            yi = rebin.rebin(y, reb)
            if reb == 7:
                noiselevel_7 = noiselevel/(7**0.5)
                #if np.sum(flag) > 0:
                tini, tfin = find_duration_simple(ti, yi, noiselevel_7, binning=7)
                #aflare = flare_class.flare(tdata=ti[tini - 10:tfin + 10]/(24*60), \
                #        ydata=yi[tini - 10:tfin + 10]/yi.max(), \
                #        noise_level=noiselevel_7/yi.max(), \
                #        yerrdata=np.zeros(tfin-tini + 20)+noiselevel_7/yi.max(), \
                #        tbeg=ti[tini]/(24.*60.), tend=ti[tfin]/(24.*60.), tpeak=0.)
                #aflare.fit_line()
                #aflare.fit_flare_profile(1, 4)
                #try:
                #    #durat_21s.append(aflare.durations[0].to(u.min).value)
                #    durat_21s.append(aflare.result_nodip.params['fwhm0'].value*24*60.)
                #except AttributeError:
                #    set_trace()
                #ampl_21s.append(aflare.result_nodip.params['ampl0']*yi.max())
                color='orange'
                #else:
                #    pass
            elif reb == 20:
                noiselevel_20 = noiselevel/(20**0.5)
                #if np.sum(flag) > 0:
                tini, tfin = find_duration_simple(ti, yi, noiselevel_20, binning=20)
                #aflare = flare_class.flare(tdata=ti[tini - 10:tfin + 10]/(24*60), \
                #        ydata=yi[tini - 10:tfin + 10]/yi.max(), \
                #        noise_level=noiselevel_20/yi.max(), \
                #        yerrdata=np.zeros(tfin-tini + 20)+noiselevel_20/yi.max(), \
                #        tbeg=ti[tini]/(24.*60.), tend=ti[tfin]/(24.*60.), tpeak=0.)
                #aflare.fit_line()
                #aflare.fit_flare_profile(1, 4, plots=True, plotname='/home/giovanni/Desktop/flaretest.pdf')
                #try:
                #    #durat_60s.append(aflare.durations[0].to(u.min).value)
                #    durat_60s.append(aflare.result_nodip.params['fwhm0'].value*24*60.)
                #except AttributeError:
                #    set_trace()
                #ampl_60s.append(aflare.result_nodip.params['ampl0']*yi.max())
                color='g'
                #else:
                #    pass
        #    print('Rebin:', reb, ' Obs peak:', yi.max())
            plt.plot(ti, yi, '+-', label=labels[i])
            plt.plot([ti[tini], ti[tini]], [yi.min(), 2.*yi.max()], '--', c=color)
            plt.plot([ti[tfin], ti[tfin]], [yi.min(), 2.*yi.max()], '--', c=color)

        plt.xlabel('Time [min]', fontsize=14)
        plt.ylabel('Relative flux', fontsize=14)
        plt.legend()
        plt.show()
        set_trace()
        #plt.close()

    plt.figure()

    fit_params_line = Parameters()
    fit_params_line.add('a', value=0., vary=False)
    fit_params_line.add('b', value=0., vary=True)
    fit_params_line.add('c', value=0., vary=True)

    plt.loglog(durat_60s, ampl_60s, '.', alpha=0.5, label='60 s')
    fit3 = minimize(models.residual_line, fit_params_line, \
        calc_covar=False, args=(np.log10(durat_60s), np.log10(ampl_60s), \
        np.zeros(len(durat_60s)) + noiselevel_20/np.array(ampl_60s)), \
        method='least_squares', nan_policy='omit')
    print('Fit 60s:', fit3.params)
    plt.loglog(durat_60s, 10**np.polyval(fit3.params, np.log10(durat_60s)), 'blue', \
            linewidth=2)

    plt.loglog(durat_21s, ampl_21s, '.', label='21 s')
    fit2 = minimize(models.residual_line, fit_params_line, \
        calc_covar=False, args=(np.log10(durat_21s), np.log10(ampl_21s), \
        np.zeros(len(durat_21s)) + noiselevel_7/np.array(ampl_21s)), \
        method='least_squares', nan_policy='omit')
    print('Fit 21s:', fit2.params)
    plt.loglog(durat_21s, 10**np.polyval(fit2.params, np.log10(durat_21s)), 'orange', \
            linewidth=2)

    plt.loglog(durat_3s, ampl_3s, '.', label='3 s')
    fit1 = minimize(models.residual_line, fit_params_line, \
        calc_covar=False, args=(np.log10(durat_3s), np.log10(ampl_3s), \
        np.zeros(len(durat_3s)) + noiselevel/np.array(ampl_3s)), \
        method='least_squares', nan_policy='omit')
    print('Fit 3s:', fit1.params)
    plt.loglog(durat_3s, 10**np.polyval(fit1.params, np.log10(durat_3s)), 'g', \
            linewidth=2)

    plt.loglog(true_durat, true_ampl, '.', label='True')
    #fit4 = np.polyfit(np.log10(true_durat), np.log10(true_ampl), \
    #        deg=1, cov=True)
    #print('No noise fit:', fit4)
    #plt.loglog(true_durat, 10**np.polyval(fit4[0], np.log10(true_durat)), 'red', \
    #        linewidth=2)

    plt.xlabel('Simulated flare FWHM [min]', fontsize=14)
    plt.ylabel('Simulated flare luminosity [erg s$^{-1}$]', fontsize=14)

    plt.legend()

    plt.savefig( \
        '/home/giovanni/Projects/CHEOPS/ancillary/shortterm_variability/plots/theoretical_lum_duration_binning.pdf')

    return

def find_duration_simple(t, y, noiselevel, binning=1):
    '''
    Including median filters and so on.
    '''
    tini = 0
    tend = 0

    windowcut = int(29/binning**0.5)
    if windowcut > 1:
        if windowcut % 2 == 0:
            windowcut += 1
        y = medfilt(y, kernel_size=windowcut)

    index = abs(t).argmin()
    i = 0
    while y[i + index] > noiselevel:
        tend += 1
        i += 1
    i = 0
    while y[i + index] > noiselevel:
        tini -= 1
        i -= 1

    return tini + index, tend + index
