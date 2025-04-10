import numpy as np
import train_utils as utils
import euclidemu2
from copy import copy
from scipy.interpolate import interp1d

import os
emulator_dir = os.path.dirname(os.path.abspath(__file__))
ks = np.loadtxt(f"{emulator_dir}/ks.txt")
log10ks = np.log10(ks)
zs_cola = [
    0.000, 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 
    0.342, 0.378, 0.417, 0.457, 0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929, 
    1.000, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 2.000, 2.158, 2.333, 2.529, 2.750, 
    3.000
]

print("[colaemu] Loading models")
models = {}
for z in zs_cola:
    models[z] = utils.load_model(f"{emulator_dir}/models/NN_Z{z:.3f}.model")
print("[colaemu] Models loaded")

ee2 = euclidemu2.PyEuclidEmulator()

# Preload constant parameters
COSMO_PARAMS_TEMPLATE = {
    'w': -1,
    'wa': 0,
    'mnu': 0.058,
}

def get_boost(x, k_custom=None):
    boosts = []
    # Precompute boost_proj_ee2 for all zs_cola
    cosmo_params = COSMO_PARAMS_TEMPLATE.copy()
    cosmo_params.update({
        'h': x[0],
        'Omega_b': x[1],
        'Omega_m': x[2],
        'As': x[3],
        'ns': x[4],
    })
    _, boost_proj_ee2 = ee2.get_boost(cosmo_params, zs_cola, ks if k_custom is None else k_custom)
    # Projecting onto LCDM
    x_proj = copy(x)
    x_proj[-1] = 0
    x_proj[-2] = -1
    if k_custom is not None:
        log10k_custom = np.log10(k_custom)
        mask_low_k = log10k_custom < log10ks[0]
    for i, (z, model) in enumerate(models.items()):
        boost_case, boost_proj = model.predict([x, x_proj])
        if k_custom is None: boost = boost_case * boost_proj_ee2[i] / boost_proj
        else:
            ratio = boost_case/boost_proj
            interp = interp1d(
                log10ks,
                ratio,
                kind='linear',
                fill_value='extrapolate', 
                assume_sorted=True
            )
            ratio_k_custom = interp(log10k_custom)
            boost = boost_proj_ee2[i] * ratio_k_custom
            boost[mask_low_k] = 1.0

        boosts.append(boost)
    return boosts