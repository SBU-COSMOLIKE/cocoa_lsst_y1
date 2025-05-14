"""
    Utility that replicates Cocoa's boost calculation outside its environment. Useful for debugging purposes (e.g. plotting the boosts)
"""

import sys
sys.path.append("emulators/joao")
import colaemu
import euclidemu2 as ee2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline

ee2 = ee2.PyEuclidEmulator()

z_interp_2D = np.linspace(0,2.0,80)
z_interp_2D = np.concatenate((z_interp_2D, np.linspace(2.01,10,20)),axis=0)
z_interp_2D[0] = 0

len_log10k_interp_2D = 1400
log10k_interp_2D = np.linspace(-4.2,2.0,len_log10k_interp_2D)

# Cobaya wants k in 1/Mpc
k_interp_2D = np.power(10.0, log10k_interp_2D)

def get_dict(params):
    return {
        'h': params[0],
        'Omb': params[1],
        'Omm': params[2],
        'As': params[3],
        'ns': params[4],
        'w': params[5],
        'wa': params[6],
        'mnu': 0.058,
    }

def get_cocoa_boost_ee2(params):
    params = get_dict(params)

    kbt = np.power(10.0, np.linspace(-2.0589, 0.973, len(k_interp_2D)))
    kbt, tmp_bt = ee2.get_boost(params, z_interp_2D, kbt)
    logkbt = np.log10(kbt)
    boosts = []
    for i in range(len(z_interp_2D)):    
        interp = interp1d(logkbt,
            np.log(tmp_bt[i]), 
            kind = 'linear', 
            fill_value = 'extrapolate', 
            assume_sorted = True
            )

        lnbt = interp(log10k_interp_2D)
        lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0
        boosts.append(np.exp(lnbt))
    return boosts

def get_cocoa_boost_colaemu(params):
    kbt = np.power(10.0, np.linspace(-2.0589, 0.973, len(k_interp_2D)))
    cola_boost = colaemu.get_boost(params, k_custom=kbt)
    cola_logboost = np.log(cola_boost)
    logkbt = np.log10(kbt)
      
    logboosts_extrap = []
    mask_low_k = log10k_interp_2D < logkbt[0]
    for i, _ in enumerate(colaemu.zs_cola):
        interp = interp1d(
            logkbt,
            cola_logboost[i],
            kind='linear',
            fill_value='extrapolate', 
            assume_sorted=True
        )
        
        logboost_extrap = interp(log10k_interp_2D)
        logboost_extrap[mask_low_k] = 0.0
        logboosts_extrap.append(logboost_extrap)
    
    logboost_2d_interp = RectBivariateSpline(colaemu.zs_cola, log10k_interp_2D, logboosts_extrap)
    logboost_2d = logboost_2d_interp(z_interp_2D, log10k_interp_2D)
    logboost_2d[z_interp_2D > 3, :] = 0.0
    return np.exp(logboost_2d)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
params = [0.67, 0.049, 0.319, 2.1e-9, 0.96, -1, 0]
for i, w0 in enumerate([-1.15, -0.85]):
    for j, wa in enumerate([-0.35, 0.25]):
        params[-1] = wa
        params[-2] = w0
        boosts_ee2 = get_cocoa_boost_ee2(params)
        boosts_cola = get_cocoa_boost_colaemu(params)
        axs[i, j].semilogx(k_interp_2D, boosts_ee2[0], label="EE2")
        axs[i, j].semilogx(k_interp_2D, boosts_cola[0], label="COLA")
        axs[i, j].set_title(f"$w_0 = {w0}$, $w_a = {wa}$")
        axs[i, j].legend()
plt.savefig("test.pdf")