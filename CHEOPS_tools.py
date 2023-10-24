'''
Code used to detrend and search for flares in CHEOPS light curves.
'''

def merge_DRT(target):
    '''
    Merge all DRT imagette light curves for a given target.
    '''
    workfolder = homedir + '/Projects/CHEOPS/ancillary/shortterm_variability/DRT/'
    lcdata = workfolder + 'DRT14_data/'
    lcfolder = workfolder + 'DRT14_results_mendoza_sympathetic/'
    lcfiles = glob.glob(lcdata + '*' + target + '_*Imagette*/' \
                + 'Mask/A_Flux_Table_Mask.fits')

    lcorder = []
    for ll in lcfiles:
        lcorder.append(ll.split('Imagette_')[1])
    order = np.array(lcorder).argsort()
    lcfiles = np.array(lcfiles)[order]
    set_trace()
    if len(lcfiles) == 0:
        return 'No subarrays'

    without = open(datadir + 'CHEOPS-ST-GTO/pycheops_data/withoutimg.txt', 'r')
    lines = without.readlines()

    # Update the first file in the list
    fields = ['MJD', 'Flux', 'FieldRot', 'FieldRotAdded', \
                'Background', 'ThermFront2']
    results = {}
    results['Visit'] = []
    results['Glint'] = []
    results['File_key'] = []
    results['Visit_tini'] = []
    # Use this to reject visit with glint alert
    results['Visit_glint'] = []
    results['Flux_abs'] = []
    nlc = 0
    results['time_on_target'] = [] # in days
    # This is useful to inspect non-folded merged LCs
    results['FieldRotAdded'] = []
    for j, lcfile in enumerate(lcfiles):
        # Avoid specific visits
        if '43702' in lcfile or '36804' in lcfile or '36902' in lcfile:
            continue
        print(lcfile)
        LC = fits.open(lcfile)
        for f in fields:
            if j == 0:
                results[f] = []
            if f != 'Flux' and f != 'FieldRotAdded':
                results[f].append(LC[1].data[f])
            elif f == 'FieldRotAdded':
                results[f].append(LC[1].data['FieldRot'] + j*360.)
            else:
                yclip = sigma_clip(LC[1].data[f], sigma=3)
                bkgclip = sigma_clip(LC[1].data['Background'], sigma=3)
                results[f].append(LC[1].data[f]/np.ma.median(yclip))
        # Mark visits
        results['time_on_target'].append(np.ptp(LC[1].data['MJD']))
        results['Flux_abs'].append(np.ma.median(yclip))
        results['Visit'].append(lcfile.split('Imagette_')[1].split('/Mask')[0])
        results['Visit_tini'].append(LC[1].data['MJD'].min())

        # Attach glint info
        filekey = lcfile.split('_Imagette_')[1].split('_V0300')[0]
        results['File_key'].append(filekey)
        if '100018' in lcfile:
            try:
                DRPfile = glob.glob('/home/giovanni/Projects/CHEOPS/ancillary/' \
                + 'shortterm_variability/DRP14/*' + filekey + '*.npz')[0]
                glint = np.load(DRPfile)['glint_flag']
            except IndexError:
                print('Failed ' + target)
                if target == 'GJ 450' or target == 'HD 265866' or target == 'GJ 2' \
                        or target == 'GJ 205':
                    glint = False
                else:
                    #set_trace()
                    return 'missing dataset'

            if glint:
                results['Glint'].append(np.ones(len(LC[1].data[f])))
                results['Visit_glint'].append(1.)
            else:
                results['Glint'].append(np.zeros(len(LC[1].data[f])))
                results['Visit_glint'].append(0.)
        else: # Retain these ones
            results['Glint'].append(np.zeros(len(LC[1].data[f])))
            results['Visit_glint'].append(0.)

    for f in fields:
        try:
            results[f] = np.hstack(results[f])
        except KeyError:
            set_trace()

    #results['Visit'] = np.hstack(results['Visit'])
    results['nvisits'] = len(lcfiles)
    results['Glint'] = np.hstack(results['Glint'])
    results['time_on_target'] = np.sum(results['time_on_target'])

    fout = open(lcfolder + target.replace(' ', '_') + '_combined.pic', 'wb')
    pickle.dump(results, fout)
    fout.close()

    return
  
def detrend_DRT(folder_data, target='', harmonics=5, time_trend=1, num=-1, \
                method='sinusoids', rotadd=False):
    '''
    Attempt for automatic detrending of DRT LCs.
    '''

    if num > -1:
        LC = glob.glob(folder + folder_data + target + '*_' + str(num) \
                + '_combined.pic')
    else:
        LC = glob.glob(folder + folder_data + target + '_combined.pic')

    # Get Teff and mag for this target
    if 'injection' not in folder_data:
        DRPfile = glob.glob(folder + 'DRP14/*' + target + '*npz')[0]
    else:
        DRPfile = glob.glob(folder + 'DRP14/*' + target.split('_n')[0] + '*npz')[0]

    DRPdata = np.load(DRPfile)
    try:
        openlc = open(LC[0], 'rb')
    except IndexError:
        if 'injection' in folder_data:
            return []

    ff = pickle.load(openlc)
    openlc.close()

    if 'injection' in folder_data:
        ff = ff['data']

    try:
        bkg = ff['Background']
    except TypeError:
        set_trace()
    t = ff['MJD']
    if not rotadd:
        phi = ff['FieldRot']
    else:
        phi = ff['FieldRotAdded']
    y = ff['Flux'] - np.median(ff['Flux'])#/ff['Flux'].max()
    visit = ff['Visit']
    visit_glint = ff['Visit_glint']
    visit_tini = ff['Visit_tini']
    glint = ff['Glint']
    nvisits = ff['nvisits']
    filekey = ff['File_key']

    # First, sigma-clip
    ksize = int(20*60/3.)
    #if ksize % 2 == 0:
    #    ksize += 1
    filt = signal.medfilt(y, kernel_size=5)
    yclip = sigma_clip(y - filt, sigma_lower=3, sigma_upper=1000.)
    bkgclip = sigma_clip(bkg, sigma=5)
    # Remove flux outliers and points with bkg > threshold
    flag = np.logical_and(~yclip.mask, ~bkgclip.mask)
    tclip = t[flag]
    yclip_ = y[flag]
    phiclip = np.radians(phi[flag])
    #phiclip[phiclip > np.pi] -= 2.*np.pi
    bkgclip = bkg[flag]
    #visitclip = visit[flag]
    glintclip = glint[flag]

    # Second clipping round
    filt = signal.medfilt(yclip_, kernel_size=31)
    yclip_s = sigma_clip(yclip_ - filt, sigma_lower=5, sigma_upper=1000.)
    # Remove flux outliers and points with bkg > threshold
    flag = ~yclip_s.mask
    tclip = tclip[flag]
    yclip = yclip_[flag]#/yclip_[flag].max()
    ypool = np.copy(yclip)
    phiclip = phiclip[flag]
    bkgclip = bkgclip[flag]
    #visitclip = visitclip[flag]
    glintclip = glintclip[flag]
    yerrclip = np.zeros(len(tclip)) + np.nanstd(yclip_ - filt)

    # If the target is AU Mic, remove data around transits
    planets = {}
    planets['b'] = {}
    planets['b']['t0'] = 2458330.39017
    planets['b']['P'] = 8.46303507
    planets['b']['duration'] = 3.54/24.
    planets['c'] = {}
    planets['c']['t0'] = 2458342.2243
    planets['c']['P'] = 18.858991
    planets['c']['duration'] = 4.5/24.

    # Target-specific data point removal
    if 'AU_Mic' in target:
        for planet in ['b', 'c']:
            t0 = planets[planet]['t0']
            P = planets[planet]['P']
            duration = planets[planet]['duration']
            phase = ((tclip + 2400000.5 - t0) % P)/P
            duration_phase = (duration + 0.01)/P
            flag = np.logical_and(phase > duration_phase, \
                        phase < 1. - duration_phase)
            tclip = tclip[flag]
            yclip = yclip[flag]
            ypool = np.copy(yclip)
            phiclip = phiclip[flag]
            bkgclip = bkgclip[flag]
            #visitclip = visitclip[flag]
            glintclip = glintclip[flag]
            filt = signal.medfilt(yclip, kernel_size=31)
            yerrclip = np.zeros(len(tclip)) + np.nanstd(yclip - filt)

    # Parts masked for specific targets, because of anomalous LC behaviour
    if 'AD_Leo' in target:
        flag = np.logical_and(phiclip > 4.8, phiclip < 5.2)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'BX_Cet' in target:
        flag = np.logical_and(phiclip > 1.5, phiclip < 5.5)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'G_32-5' in target:
        flag = np.logical_or(np.logical_and(phiclip > 2.3, phiclip < 4.5), \
                    np.logical_and(phiclip > 0.4, phiclip < 0.7))
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'GJ_422' in target:
        flag = np.logical_or.reduce((np.logical_and(phiclip > 3.5, phiclip < 5), \
                    np.logical_and(phiclip > 1.7, phiclip < 2.3), \
                    np.logical_and(phiclip > 2.9, phiclip < 3.1)))
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'GJ_176' in target:
        flag = np.logical_and(phiclip > 0.95, phiclip < 1.4)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'GJ_273' in target:
       flag = np.logical_or(np.logical_and(phiclip > 0.5, phiclip < 0.9), \
                    np.logical_and(phiclip > 5.3, phiclip < 5.7))
       tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'VV_Lyn' in target:
       flag = np.logical_or(np.logical_and(phiclip > 0.3, phiclip < 1.), \
                    np.logical_and(phiclip > 5.3, phiclip < 5.6))
       tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'V_1054_Oph' in target:
       flag = np.logical_or(np.logical_and(phiclip > 1., phiclip < 1.4), \
                    np.logical_and(phiclip > 5.7, phiclip < 6.3))
       tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'GJ_9404' in target:
        flag = np.logical_and(phiclip > 0.8, phiclip < 3.6)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'MCC_549' in target:
        flag = np.logical_and(phiclip > 0., phiclip < 1.)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif 'G_168-31' in target:
        flag = np.logical_and(phiclip > 0.9, phiclip < 3.9)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)
    elif '2MASS_J13314666+2916368' in target:
        flag = np.logical_and(phiclip > 0., phiclip < 1.5)
        tclip, yclip, ypool, phiclip, bkgclip, glintclip, yerrclip \
                    = mask_DRT(flag, tclip, yclip, phiclip, bkgclip, glintclip, yerrclip)

    # Cut data right before and after each gap, there are often
    # weird edge effects
    reject = []
    gaps_visit = np.where(np.diff(tclip) > abs(np.diff(tclip)).min()*10)[0]
    if len(gaps_visit) > 0:
        tclip = remove_points_edges(tclip, gaps_visit, 20)
        yclip = remove_points_edges(yclip, gaps_visit, 20)
        ypool = remove_points_edges(ypool, gaps_visit, 20)
        phiclip = remove_points_edges(phiclip, gaps_visit, 20)
        bkgclip = remove_points_edges(bkgclip, gaps_visit, 20)
        glintclip = remove_points_edges(glintclip, gaps_visit, 20)
        yerrclip = remove_points_edges(yerrclip, gaps_visit, 20)

    # Subtract time trend for a single long visit
    if 'injection' not in folder_data:
        pplot = PdfPages(LC[0].replace('.pic', '_detrend.pdf'))
    else:
        pplot = PdfPages(LC[0].replace('data_CHEOPS', 'results_CHEOPS').replace( \
                    '.pic', '_detrend.pdf'))

    #if time_trend == 1: # Remove time trends per visit
    print('Per-visit time detrending...')
    gaps = np.where(np.diff(tclip) > 0.5)[0]
    if len(gaps) == 0:
        gaps = [-1]

    for k in np.arange(len(gaps) + 1):
        fig, axs = plt.subplots(figsize=(12, 5))
        if np.array(gaps == [-1]).all() and k == 1:
            continue
        if 'GJ_65' in target:# or 'GJ_422' in target:
            degmax = 10
        else:
            degmax = 3

        if k == 0:
            tg = tclip[:gaps[k]]
            yg = yclip[:gaps[k]]
            yerrg = yerrclip[:gaps[k]]
            axs.plot(tg, yg, '.', alpha=0.3, label='Raw data')
            aic, best_deg, continuum \
                 = pycheops_test.iterative_polyfit(tg, yg, yerrg, degmax=degmax)
            yclip[:gaps[k]] -= continuum
        elif k > 0 and k < len(gaps):
            tg = tclip[gaps[k - 1] + 1: gaps[k]]
            yg = yclip[gaps[k - 1] + 1: gaps[k]]
            yerrg = yerrclip[gaps[k - 1] + 1: gaps[k]]
            axs.plot(tg, yg, '.', alpha=0.3, label='Raw data')
            aic, best_deg, continuum \
                = pycheops_test.iterative_polyfit(tg, yg, yerrg, degmax=degmax)
            yclip[gaps[k - 1] + 1: gaps[k]] -= continuum
        else:
            tg = tclip[gaps[-1] + 1:]
            yg = yclip[gaps[-1] + 1:]
            yerrg = yerrclip[gaps[-1] + 1:]
            axs.plot(tg, yg, '.', alpha=0.3, label='Raw data')
            aic, best_deg, continuum \
                = pycheops_test.iterative_polyfit(tg, yg, yerrg, degmax=degmax)
            yclip[gaps[-1] + 1:] -= continuum
        if len(t) <= 1:
            continue
        axs.plot(tg, yg - 0.005, '.', label='Detrended')
        axs.legend()
        axs.set_ylabel('Relative flux + C', fontsize=14)
        axs.set_xlabel('Time [days]', fontsize=14)
        axs.set_title(filekey[k], fontsize=14)
        pplot.savefig(fig)

    # To get uncertainty
    filt = signal.medfilt(yclip, kernel_size=31)
    err = np.zeros(len(yclip)) + np.std(yclip - filt)

    # Detrend each visit by each background
    #for n in np.arange(nvisits):
    #    flag = visitclip == n
    #    fit = np.polyfit(bkgclip, yclip, 1)
    #    print(fit)
    #    yclip -= np.polyval(fit, bkgclip)

    print('Roll-angle detrending...')
    rej = 10
    #if num < 0:
    #    clipflag = glintclip == 0
    #else:
    clipflag = np.array([True]*len(glintclip))
    yclipsave = np.copy(yclip)
    # This will be useful to estimate the significance of a dip
    lightc = lc_obs.LC(t=tclip, y=yclip, yerr=yerrclip)
    #bins, red, white = lightc.correlated_noise(1800./86400., plots=False)
    # If there's at least one visit without glint flag
    if method == 'sinusoids' and np.sum(clipflag) > 100:
        if np.sum(clipflag) > 0:
            phitemp = np.copy(phiclip[clipflag])
            ttemp = np.copy(tclip[clipflag])
            ytemp = np.copy(yclip[clipflag])
            errtemp = np.copy(err[clipflag])
            bkgtemp = np.copy(bkgclip[clipflag])
            #visit_temp = np.copy(visitclip[clipflag])
            glint_temp = np.copy(glintclip[clipflag])
            params = Parameters()
            params.add('a0', value=0., min=-1., max=1., vary=False)
            for j in np.arange(1, harmonics + 1):
                params.add('a' + str(j), value=np.std(yclip))#, min=0., max=10.)
                params.add('b' + str(j), value=np.std(yclip))#, min=0., max=10.)
            params.add('d', value=0., vary=False)
            params.add('e', value=0., vary=False)
    
            # Avoid rejecting more than 25% of the data points
            while rej > 0:#and rej < int(len(tclip)*0.25):
                result = minimize(pycheops_test.sin_residuals_DRT, params, \
                  args=(ttemp, phitemp, ytemp, errtemp, bkgtemp, \
                  nvisits, harmonics), method='leastsq', calc_covar=False)
                model = ytemp - result.residual*errtemp
                # Find residuals
                ys = sigma_clip(ytemp - model, sigma=3)
                rej = np.sum(ys.mask == True)
                ytemp = ytemp[~ys.mask]
                ttemp = ttemp[~ys.mask]
                phitemp = phitemp[~ys.mask]
                errtemp = errtemp[~ys.mask]
                bkgtemp = bkgtemp[~ys.mask]
                #visit_temp = visit_temp[~ys.mask]
                glint_temp = glint_temp[~ys.mask]
            sm = smooth(ytemp - model, window_len=5)[:len(ytemp)]
            #smint = interpolate.interp1d(ttemp, sm, fill_value='extrapolate', \
            #            bounds_error=False)
            #model2 = smint(tclip)
            err = np.zeros(len(tclip)) + np.nanstd(ytemp - model - sm)
            #err = np.zeros(len(tclip)) + np.nanstd(sm)
            ytempsave = np.copy(ytemp - model)
            model = yclip - pycheops_test.sin_residuals_DRT(result.params, \
                tclip, phiclip, yclip, err, bkgclip, nvisits, \
                harmonics, plots=False)*err
            yclip -= model
        else:
            model = np.zeros(len(yclip)) + np.nanmedian(yclip)
            ytemp = np.copy(yclip)
            ttemp = np.copy(tclip)
            phitemp = np.copy(phiclip)
            glint_temp = np.copy(glintclip)
            sm = smooth(ytemp - model, window_len=5)[:len(ytemp)]
            #err = np.zeros(len(tclip)) + np.nanstd(ytemp - model - sm)
            err = np.nanstd(sm)
            #smint = interpolate.interp1d(ttemp, sm, fill_value='extrapolate', \
            #            bounds_error=False)
            #model2 = smint(tclip)

    elif method == 'roll_smooth':# and np.sum(clipflag) > 100:
        order = phiclip.argsort()
        phiorder = phiclip[order]
        clipflagorder = clipflag[order]
        # Save original yclip order
        ysave = [(value, index) for index, value in enumerate(order)]

        yorder = yclip[order]
        ytemp = yclip[order][clipflag]
        phitemp = np.copy(phiorder[[clipflag]])
        rej = 100
        while rej > 0 and rej < int(len(tclip)*0.25):
            #model = signal.medfilt(ytemp, kernel_size=51)
            try:
                model = smooth(ytemp, window_len=1001)[:len(ytemp)]
            except ValueError:
                model = smooth(ytemp, window_len=int(len(ytemp)/3.))[:len(ytemp)]
            ys = sigma_clip(ytemp - model, sigma=3)
            rej = np.sum(ys.mask == True)
            phitempsave = np.copy(phitemp)
            ytemp = ytemp[~ys.mask]
            phitemp = phitemp[~ys.mask]
        model_int = interpolate.interp1d(phitempsave, model, \
                bounds_error=False, fill_value='extrapolate')
        model = model_int(phiorder)
        yorder -= model
        # Back to time order
        yclip = np.zeros(len(yorder))
        for i in ysave:
            yclip[i[0]] = yorder[i[1]]
        sm = smooth(yclip, window_len=11)[:len(yclip)]
        err = np.zeros(len(tclip)) + np.nanstd(yclip - sm)
    elif method == 'GP' and np.sum(clipflag) > 0:
        order = phiclip.argsort()
        phiorder = phiclip[order]
        clipflagorder = clipflag[order]
        # Save original yclip order
        ysave = [(value, index) for index, value in enumerate(order)]

        yorder = yclip[order]
        ytemp = yclip[order][clipflag]
        errtemp = err[order]
        phitemp = np.copy(phiorder[clipflag])
        kernel = terms.SHOTerm(sigma=np.std(ytemp), w0=1., Q=1.)
        #kernel = terms.Matern32Term(sigma=np.std(ytemp), rho=np.pi)
        gp = celerite2.GaussianProcess(kernel, mean=np.median(ytemp))
        gp.compute(phitemp, yerr=errtemp)
        initial_params = [np.median(ytemp), np.std(ytemp), 2., 1., 1.]

        #initial_params = [np.median(ytemp), np.std(ytemp), np.pi, 1.]
        #bounds = [(-1, 1.), (1e-6, 10.), (np.radians(60.), 2.*np.pi), (1., 2.)]
        while rej > 0:
            bounds = [(-1., 1.), (1e-6, np.std(ytemp)), (0.1, 10.), (np.radians(60.), 2.*np.pi), (1., 2.)]
            soln = optimize.minimize(neg_log_like, initial_params, \
                    bounds=bounds, method="L-BFGS-B", args=(gp, phitemp, ytemp, errtemp))
            opt_gp = set_params(soln.x, gp, phitemp, errtemp)
            gp.compute(phitemp, yerr=errtemp)
            model = opt_gp.predict(ytemp, phitemp, return_var=False)
            ys = sigma_clip(ytemp - model, sigma=3)
            rej = np.sum(ys.mask == True)
            phitempsave = np.copy(phitemp)
            ytemp = ytemp[~ys.mask]
            phitemp = phitemp[~ys.mask]
            errtemp = errtemp[~ys.mask]
            #print(rej)
        err = np.zeros(len(tclip)) + np.nanstd(ytemp - model)

        # Predict for the initial data set --> missing values
        model = opt_gp.predict(ytemp, phiorder)
        #_, var = opt_gp.predict(ytemp, phiorder[::10], return_var=True)
        #varint = interpolate.interp1d(phiorder[::10], var, \
        #        bounds_error=False, fill_value='extrapolate')
        #var = varint(phiorder)
        #stddev = var**0.5
        yorder -= model
        #stddev = np.nanstd(yorder)
        # Back to time order
        yclip = np.zeros(len(yorder))
        for i in ysave:
            yclip[i[0]] = yorder[i[1]]
    else:
        model = np.median(yclip)
        sm = smooth(yclip, window_len=11)[:len(yclip)]
        err = np.zeros(len(tclip)) + np.nanstd(yclip - sm)

    plt.close('all')

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    #if method != 'time_smooth':
        #axs[0].plot(phiclip[glintclip == 0.], yclipsave[glintclip == 0.], '.')
    if method == 'sinusoids':
        axs[0].plot(phiclip, yclipsave, 'b.', alpha=0.2)
        axs[0].plot(phitemp, ytemp, 'b.')
        axs[0].plot(phiclip, model, 'r.')
    #axs[0].plot(phiclip[glintclip == 1.], yclipsave[glintclip == 1.], 'r.', \
    #        label='Glint flag')
    elif method == 'roll_smooth' and np.sum(clipflag) > 100:
        axs[0].plot(phitemp, ytemp, 'b.')
        axs[0].plot(phiorder, yclip[order], 'b.', alpha=0.2)
        axs[0].plot(phiorder, model, 'orange')
    elif method == 'GP':
        axs[0].plot(phiclip, yclipsave[order], 'b.', alpha=0.2, label='Excluded')
        axs[0].plot(phitemp, ytemp, 'b.', label='Retained')
        axs[0].plot(phiorder, model, 'r', label='Model')
        #axs[0].fill_between(phiorder, y1=model + stddev, \
        #            y2=model - stddev, color='orange')
    axs[0].legend()

    if method != 'time_smooth':
        axs[0].set_xlabel('Roll angle [rad]', fontsize=14)
    else:
        axs[0].set_xlabel('Relative time [days]', fontsize=14)
    axs[0].set_ylabel('Relative flux', fontsize=14)
    #axs[1].plot(bkgclip, yclip, '.')
    axs[1].plot(yclipsave, '.')
    #mm = np.zeros(len(model))
    #for i, j in enumerate(model):
    #    mm[i] = model[order[i]]
    axs[1].plot(yclip, '.')
    #axs[1].set_xlabel('Background [e$^{-}$?]', fontsize=14)
    axs[1].set_xlabel('Time without gaps [days]', fontsize=14)
    pplot.savefig(fig)

    # Pooled variance
    vari = []
    pooled_variance = []
    poolok = []

    if np.diff(tclip).min() < 4./86400.:
        # From 9 seconds to 90 minutes, with 3 sec cadence
        pools = np.arange(3, 90*60/3, 3)
        for j in pools:
            slice_edges = np.arange(len(tclip))[::int(j)]
            sliceok = 0
            for k, se in enumerate(slice_edges):
                if k == 0:
                    continue
                elif k == len(slice_edges) - 1:
                    yslice = (ypool - model)[se:]**2
                    sliceok += 1
                else:
                    yslice = (ypool - model)[slice_edges[k - 1]:slice_edges[k]]**2
                if len(yslice) > 0:
                    vari.append(np.var(yslice))
                    #bins = np.diff(slice_edges)[0]
                    sliceok += 1
            if sliceok == 0:
                continue
            else:
                poolok.append(j)
                pooled_variance.append(np.sum(vari)/sliceok)

        fig2, axs2 = plt.subplots(figsize=(12, 5), ncols=2)
        axs2[0].semilogx(np.array(poolok)*3/60., np.array(pooled_variance)*1e8)
        axs2[0].set_xlabel('Bin size [min]', fontsize=14)
        axs2[0].set_ylabel(r'Pooled variance $\times 10^8$ [ppm$^2$]', fontsize=14)
        axs2[1].loglog(np.array(poolok)*3/60., np.array(pooled_variance)*1e8)
        axs2[1].set_xlabel('Bin size [min]', fontsize=14)
        pplot.savefig(fig2)
    pplot.close()

    # Compute red noise level on 3-sigma-clipped version of LC (this should
    # remove flares)
    #yclip2 = sigma_clip(yclip, sigma=3)
    lightc = lc_obs.LC(t=ttemp, y=ytempsave)
    bins, red, white = lightc.correlated_noise(3600./86400., \
            interval=20, plots=False)
    # When the bin is longer than the LC?
    flag = red == 0.
    bins = bins[~flag]
    red = red[~flag]
    white = white[~flag]

    ff['Flux_err'] = err #np.zeros(len(err)) + np.median(DRPdata['yerr'])*14**0.5
    ff['Flux_detrended'] = yclip
    ff['Red_noise'] = [bins, red, white]
    ff['Background_clip'] = bkgclip
    ff['MJD_clip'] = tclip
    ff['FieldRot_clip'] = phiclip
    ff['Visit_clip'] = np.array(visit)#[np.array(visit_glint) == 0.]
    ff['Glint_clip'] = glintclip
    ff['Visit_tini_clip'] = np.array(visit_tini)#[np.array(visit_glint) == 0.]
    #ff['nvisits_clip'] = visitclip
    if np.diff(tclip).min() < 4./86400.:
        ff['Pooled_variance'] = [pools*3/60, np.array(pooled_variance)*1e8]
    ff['gmag'] = DRPdata['gmag']

    if 'injection' not in folder_data:
        openlc = open(LC[0], 'wb')
    else:
        openlc = open(LC[0].replace('data_CHEOPS', 'results_CHEOPS'), 'wb')
    pickle.dump(ff, openlc)
    openlc.close()

    return err

def search_flares_DRT(folder_data, filt_kernel_size, tgstring='', num=-1, \
            mode='poly', norm='poly', resultfolder=None, rebin=0):
    '''
    Look for flares on reduced LCs
    '''

    saveresfolder = folder + folder_data
    if num > -1:
        LC = glob.glob(saveresfolder + '*' + tgstring + '_' + str(num) + \
                '_combined.pic')[0]
    else:
        try:
            LC = glob.glob(saveresfolder.replace('_onemin', \
                    '').replace('_21sec', '') + '*' \
                    + tgstring + '_combined.pic')[0]
        except IndexError:
            print('No light curves for this target.')
            #set_trace()
            return
    wth, fth = np.loadtxt(throughput_folder + 'CHEOPS_CHEOPS.band.dat', \
                            unpack=True)

    good = 0

    if mode == 'preselection':
        keys = []
        # Previously found flares for a given target
        found_files = glob.glob(saveresfolder + '*' + tgstring \
                    + '_combined_flare_results.pic')
        try:
            found = pickle.load(open(found_files[0], 'rb'))
        except IndexError:
            set_trace()
        peaki = found[tgstring].keys()

        for p in peaki:
            set_trace()
            keys.append(found[tgstring][p][0].file_key)

    dict_sptypes = {}
    dict_sptypes['K5V'] = 4450
    dict_sptypes['K6V'] = 4200
    dict_sptypes['K7V'] = 4050
    dict_sptypes['K8V'] = 3970
    dict_sptypes['K9V'] = 3880
    dict_sptypes['M0V'] = 3850
    dict_sptypes['M1V'] = 3680
    dict_sptypes['M2V'] = 3550
    dict_sptypes['M3V'] = 3400
    dict_sptypes['M4V'] = 3200
    dict_sptypes['M5V'] = 3050
    dict_sptypes['M6V'] = 2800

    results = {}

    results[tgstring] = {}

    if 'injection' not in folder_data:
        target = LC.split('results_mendoza_sympathetic/')[1].split('_combined')[0]
    else:
        if 'onemin' not in LC and '21sec' not in LC:
            target = LC#LC.split('results_CHEOPS/')[1].split('_combined')[0]
        else:
            target = LC.split('results_CHEOPS_onemin/')[1].split('_combined')[0]

    ff = pickle.load(open(LC, 'rb'))
    ttot = ff['MJD_clip']
    # Use only LCs with short-cadence data
    #if np.diff(ttot).min() > 22./86400.:
    #    continue
    ytot = ff['Flux_detrended']
    yerrtot = ff['Flux_err']
    rednoise = ff['Red_noise']

    if rebin > 1:
        light_curve = lc_obs.LC(t=ttot, y=ytot, yerr=yerrtot)
        ttot, ytot, yerrtot = light_curve.rebin(rebin)
        fout = open(saveresfolder + tgstring + '_binned_LC.pic', 'wb')
        pickle.dump([ttot, ytot, yerrtot], fout)

    # Rearrange
    order = ttot.argsort()
    ttot = ttot[order]
    ytot = ytot[order]
    order2 = np.array(ff['Visit_tini']).argsort()
    try:
        vistot = ff['Visit_clip'][order2]
        vistot_ini = np.array(ff['Visit_tini'])[order2]
        glinttot = np.array(ff['Visit_glint'])[order2]
        yabstot = np.array(ff['Flux_abs'])[order2]
    except IndexError:
        vistot = ff['Visit_clip']
        vistot_ini = ff['Visit_tini']
        glinttot = np.array(ff['Visit_glint'])
        yabstot = np.array(ff['Flux_abs'])
    dftempnew = pd.read_csv(folder + 'additional_data/' \
            + 'calibTeffs_pastel.csv')
    row = dftempnew['Target'] == target

    try:
        tstar = int(dftempnew[row]['calibTeff'].values[0])
    except IndexError:
        if 'injection' not in folder_data:
            target_data = np.load(glob.glob(folder + 'DRP14/*' \
                    + tgstring + '*npz')[0], allow_pickle=True)
        else:
            target_data = np.load(glob.glob(folder + 'DRP14/*' \
                    + tgstring.split('_n')[0] + '*npz')[0], allow_pickle=True)
        if target == 'EQ_Peg':
            tstar = dict_sptypes['M4V']
        elif 'AU_Mic' in target:
            tstar = dict_sptypes['M1V']
        elif 'GJ_740' in target:
            tstar = 3584.
        elif 'GJ_588' in target:
            tstar = 3430.
        else:
            try:
                tstar = dict_sptypes[target_data['sptype'].item()]
            except KeyError:
                if 'GJ_701' in target:
                    tstar = dict_sptypes['M0V']
                elif 'GJ_70' in target:
                    tstar = dict_sptypes['M1V']
    try:
        rstar = np.round(phot_analysis.teff_radius_calib(tstar), 2)
    except UnboundLocalError:
        set_trace()

    # Divide LCs in segments, plot them all
    gaps = np.where(np.diff(ttot) > 0.5)[0]
    if len(gaps) == 0:
        gaps = [-1]

    if 'injection' not in folder_data:
        plotname_segments = LC.split('DRT')[0] + folder_data + LC.split( \
            '/')[-1].replace('.pic',  '_segments.pdf')
    else:
        #plotname_segments = LC.replace('_combined.pic', '_segments.pdf')
        plotname_segments = saveresfolder + LC.split('_CHEOPS/')[1].replace( \
                'combined.pic', 'segments.pdf')
    pplot = PdfPages(plotname_segments)
    #pplot = PdfPages(LC.replace('.pic', '_segments.pdf'))
    fig, axs = plt.subplots(figsize=(20, 12))
    for k in np.arange(len(gaps) + 1):
        if np.array(gaps == [-1]).all() and k == 1:
            continue
        if k == 0:
            t = ttot[:gaps[k]]
            y = ytot[:gaps[k]]
            yerr = yerrtot[:gaps[k]]
        elif k > 0 and k < len(gaps):
            t = ttot[gaps[k - 1] + 1: gaps[k]]
            y = ytot[gaps[k - 1] + 1: gaps[k]]
            yerr = yerrtot[gaps[k - 1] + 1: gaps[k]]
        else:
            t = ttot[gaps[-1] + 1:]
            y = ytot[gaps[-1] + 1:]
            yerr = yerrtot[gaps[-1] + 1:]
        if len(t) <= 1:
            continue
        t = (t - t.min())*24.*60.
        axs.errorbar(t, y + 0.02*k, yerr=yerr, fmt='.', errorevery=1000)
    axs.set_title(tgstring.replace('_', ' '), fontsize=24)
    axs.set_xlabel('t - t$_0$ [min]', fontsize=20)
    axs.set_ylabel('Relative flux + constants', fontsize=20)
    for xy in ['x', 'y']:
        axs.tick_params(axis=xy, labelsize=14)
    pplot.savefig(fig)

    if 'AU_Mic' in tgstring:
        results['AU_Mic'] = []
    else:
        results[tgstring] = []
    for k in np.arange(len(gaps) + 1):
        if np.array(gaps == [-1]).all() and k == 1:
            continue
        if k == 0:
            t = ttot[:gaps[k]]
            y = ytot[:gaps[k]]
            yerr = yerrtot[:gaps[k]]
            glint = glinttot[k]
            if len(t) <= 1:
                continue
        elif k > 0 and k < len(gaps):
            t = ttot[gaps[k - 1] + 1: gaps[k]]
            y = ytot[gaps[k - 1] + 1: gaps[k]]
            yerr = yerrtot[gaps[k - 1] + 1: gaps[k]]
            if len(t) <= 1:
                continue
        else:
            t = ttot[gaps[-1] + 1:]
            y = ytot[gaps[-1] + 1:]
            yerr = yerrtot[gaps[-1] + 1:]
            if len(t) <= 1:
                continue

        filekey = vistot[abs(vistot_ini - t.min()).argmin()]
        #if '29708' in filekey or '35702' in filekey:
        #    continue
        print('File key:', filekey)
        glint = glinttot[abs(vistot_ini - t.min()).argmin()]
        #if glint == 1.:
        #    continue

        # Some light curves deserve specific treatment
        if 'injection' not in folder_data:
            t -= t.min()
        if norm == 'poly':
            # Normalize with polynomial, attempt different degrees
            aic = np.inf
            if '100018' in filekey:
                dmax = 3
            elif '100010' in filekey:
                dmax = 10
            for d, degree in enumerate(np.arange(dmax)):
                nrej = 10
                if 'PR100018_TG036307' in filekey:
                    tflag = np.logical_or(t < 0.2/(24*60.), \
                                t > 60./(24.*60.))
                    ttemp = t[tflag]
                    ytemp = y[tflag]
                    yerrtemp = yerr[tflag]
                elif 'PR100018_TG053201' in filekey:
                    tflag = t > 60./(24.*60.)
                    ttemp = t[tflag]
                    ytemp = y[tflag]
                    yerrtemp = yerr[tflag]
                elif '35202' in filekey:
                    tflag = np.logical_or(t < 0.020, t > 0.039)
                    ttemp = t[tflag]
                    ytemp = y[tflag]
                    yerrtemp = yerr[tflag]
                elif '45203' in filekey:
                    tflag = np.logical_or(t < 220/(24*60.), \
                                                t > 350./(24.*60.))
                    ttemp = t[tflag]
                    ytemp = y[tflag]
                    yerrtemp = yerr[tflag]
                elif '30403' in filekey:
                    tflag = np.logical_or(t < 15/(24*60.), \
                                                t > 40./(24.*60.))
                    ttemp = t[tflag]
                    ytemp = y[tflag]
                    yerrtemp = yerr[tflag]
                else:
                    ttemp = np.copy(t)
                    ytemp = np.copy(y)
                    yerrtemp = np.copy(yerr)
                iter = 0
                while nrej > 0 or iter == 100:
                    pars = Parameters()
                    for i in np.arange(int(degree)):
                        pars.add('c' + str(int(i)), value=-0.1, vary=True)
                    result = minimize(pycheops_test.poly_residuals, \
                       pars, args=(ttemp, ytemp, yerrtemp), nan_policy='omit', \
                       calc_covar=False, method='powell')
                    continuum_d = np.polyval(result.params, ttemp)
                    sc = sigma_clip(ytemp - continuum_d, sigma=3)
                    ttemp = ttemp[~sc.mask]
                    ytemp = ytemp[~sc.mask]
                    yerr_flare = np.nanstd(sc.data[~sc.mask])
                    yerrtemp = yerrtemp[~sc.mask]
                    nrej = np.sum(sc.mask)
                    iter += 1
                aic_d = result.aic
                if aic_d < aic:
                    aic = aic_d
                    best_deg = degree
                    best_result = copy.deepcopy(result)
                    window = int(30*60/3.)
                    continuum = np.polyval(result.params, t)
                    filteredLC = y - continuum
                    ttempsave = np.copy(ttemp)
                    ytempsave = np.copy(ytemp)
                    yerr = np.zeros(len(filteredLC)) + yerr_flare

        elif norm == 'smooth':
            nrej = 10
            ttemp = np.copy(t)
            ytemp = np.copy(y)
            yerrtemp = np.copy(yerr)
            while nrej > 0:
                wl = min([int(len(ytemp)/5.), int(30.*60/3)])
                continuum = smooth(ytemp, window_len=wl)[:len(ytemp)]
                sc = sigma_clip(ytemp - continuum, sigma=3.)
                nrej = np.sum(sc.mask)
                ttemp = ttemp[~sc.mask]
                ytemp = ytemp[~sc.mask]
                yerr_flare = np.nanstd(sc.data[~sc.mask])
            # Interpolate continuum
            cont_int = interpolate.interp1d(ttemp, continuum, \
                        bounds_error=False, fill_value='extrapolate')
            continuum = cont_int(t)
            filteredLC = y - continuum
            yerr = np.zeros(len(filteredLC)) + yerr_flare
        elif norm == 'edges':
            tc = np.concatenate((t[:10], t[-10:]))
            yc = np.concatenate((y[:10], y[-10:]))
            continuum_fit = np.polyfit(tc, yc, 1)
            continuum = np.polyval(continuum_fit, t)
            filteredLC = y - continuum
            yerr_flare = np.nanmedian(yerr)
        elif norm == 'nada':
            continuum = np.zeros(len(t))
            filteredLC = np.copy(y)
            yerr_flare = np.nanmedian(yerr)

        if mode == 'preselection':
            peaki = []
            if filekey in keys:
                for j in found[tgstring].keys():
                    filek = found[tgstring][j][0].file_key
                    if filekey == filek:
                        peaki.append(j)
        elif mode == 'manual':
            peaki = []
            points = []
            figx, axsx = plt.subplots(figsize=(12, 5))
            axsx.errorbar((t - t.min())*24.*60., filteredLC, yerr=yerr, mt='.')
            axsx.set_xlabel('Time [min]', fontsize=14)
            plt.show()
            print('Select flare peaks (LEFT to add, RIGHT to remove, MIDDLE' \
                    ' to stop)')
            points = plt.ginput(100, timeout=-1)
            peaki = []
            for p in points:
                # Find flux max around selection
                peaki_ = abs((t - t.min())*24.*60. - p[0]).argmin()
                peakflux = filteredLC[peaki_ - 30 : peaki_ + 30].max()
                peaki.append(peaki_)
            goodones = len(peaki)
        else:
            peaki = []

        fig2, axs2 = plt.subplots(figsize=(12, 5))
        axs2.plot(t*24.*60., y, '.', label='DRT LC, glint ' + str(int(glint)))
        #axs2.plot(ttemp*24.*60, ytemp, '.', label='Quiet stellar flux')
        #axs2.plot(t*24.*60, continuum, label='Quiet stellar flux model')
        for p in peaki:
            axs2.plot([t[p]*24*60., t[p]*24*60.], [y[p] + 5*yerr.max(), \
                    y[p] + 10*yerr.max()], 'r', linewidth=2)
        #axs2.plot(t*24*60, continuum + 4.*yerr_flare, 'r--', \
        #            label='Flare detection threshold')
        if norm == 'poly':
            axs2.set_title(str(filekey) + ', best polynomial degree: ' \
                + str(best_deg), fontsize=14)
        elif norm == 'smooth':
            axs2.set_title(str(filekey) + ', smoothing', fontsize=14)
        axs2.set_xlabel('Time [min]', fontsize=14)
        axs2.set_ylabel('Relative flux', fontsize=14)
        axs2.set_title(filekey, fontsize=14)
        axs2.legend()
        plt.tight_layout()
        pplot.savefig(fig2)

        if 'injection' not in folder_data:
            plotname = LC.split('DRT')[0] + folder_data + LC.split( \
                '/')[-1].replace('.pic',  '_' + str(filekey) + '.pdf')
        else:
            #plotname = LC.replace('_combined.pic', '_flares.pdf')
            plotname = saveresfolder + LC.split('_CHEOPS/')[1].replace( \
                        'combined.pic', 'flares.pdf')
        flarespar = fd.find_flares([t, filteredLC, yerr], \
            peaki=peaki, flare_threshold=3., stdflare=yerr_flare, \
            force_fit=False, fit_continuum=1, normalise=False, \
            clip=True, wth=wth, fth=fth, smoothmode=None, complexity=5, \
            saveplots=plotname,
            #LC.replace('.pic', '_' + str(filekey) + '.pdf'), \
            plot_flat=False, tstar=tstar, rstar=rstar, rednoise=rednoise, \
            min_datapoints=4, outlier_thresh=8., filt_kernel_size=filt_kernel_size)

        for fp in flarespar:
            fp.filekey = filekey

        if 'AU_Mic' in tgstring:
            tgstring = 'AU_Mic'
        results[tgstring].append(flarespar)


    if len(results[tgstring]) > 0:
        results[tgstring] = np.hstack(results[tgstring])
    else:
        results[tgstring] = np.array([])
    pplot.close()
    plt.close('all')

    if resultfolder == None:
        fout = open(LC.split('DRT')[0] + folder_data + LC.split( \
                '/')[-1].replace('.pic', '_flare_results.pic'), 'wb')
        #fout = open(LC.replace('.pic', '_flare_results.pic'), 'wb')
    else:
        fout = open(LC.split('flare_inj')[0] + resultfolder \
                + LC.split('/')[-1].replace('.pic', '_flare_results.pic'), 'wb')
    pickle.dump(results, fout)
    fout.close()

    return
