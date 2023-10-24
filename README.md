# flarefinder
Code used for flare detecion and profile fitting in photometric time series. Tested on CHEOPS 3-s and TESS 20-s light curves.
If you use this code, please reference this paper: ...

The code contains functions to remove low-frequency variability, such as the one due to stellar activity, detrend CHEOPS imagette light curves, detect flare peak candidates, and fit their profiles using previously published codes and mdoels. Flare luminosity and energy calculation is performed following [Raetz et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..22R/abstract), [Davenport et al. (2014)](https://ui.adsabs.harvard.edu/abs/2014ApJ...797..122D/abstract) and [Gunther et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020AJ....159...60G/abstract)'s prescriptions. For all these steps, several options are available and can be chosen with function keywords.

Among the software which makes an essential part of the algorithms:
- Lucas Hermann Negri's [peakutils](https://pypi.org/project/PeakUtils/)
- Guadaluper Tovar Mendoza's [Llamaradas Estelares](https://github.com/lupitatovar/Llamaradas-Estelares).
