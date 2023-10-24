# flarefinder
Code used for flare detecion and profile fitting in photometric time series. Tested on CHEOPS 3-s and TESS 20-s light curves.
If you use this code, please reference this paper: ...

The code contains functions to remove low-frequency variability, such as the one due to stellar activity, detect flare peak candidates, and fit their profiles using previously published codes and mdoels. Flare luminosity and energy calculation can be performed following Raetz et al. (2020), Davenport et al. (2014) and Gunther et al. (2020) prescriptions. For all these steps, several options are available and can be chosen with function keywords.

Among the software which makes an essential part of the algorithms:
- Peakutils (https://pypi.org/project/PeakUtils/)
- Llamaradas Estelares (https://github.com/lupitatovar/Llamaradas-Estelares.
