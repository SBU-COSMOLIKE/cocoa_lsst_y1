import numpy as np
import train_utils as utils
import euclidemu2 as ee2
from copy import copy

import os
current_dir = os.getcwd() # Should be cocoa/Cocoa/
ks = np.loadtxt(f"{current_dir}/projects/lsst_y1/emulators/joao/ks.txt")
zs_cola = [
    0.000, 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 
    0.342, 0.378, 0.417, 0.457, 0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929, 
    1.000, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 2.000, 2.158, 2.333, 2.529, 2.750, 
    3.000
]

print("[colaemu] Loading models")
models = {}
for z in zs_cola:
    models[z] = utils.load_model(f"{current_dir}/projects/lsst_y1/emulators/joao/models/NN_Z{z:.3f}.model")
print("[colaemu] Models loaded")

def get_boost(x):
    boosts = []
    _, boost_proj_ee2 = ee2.get_boost({
        'h': x[0],
        'Omega_b': x[1],
        'Omega_m': x[2],
        'As': x[3],
        'ns': x[4],
        'w': -1,
        'wa': 0,
        'mnu': 0.058,
    }, zs_cola, ks)
    # Projecting onto LCDM
    x_proj = copy(x)
    x_proj[-1] = 0
    x_proj[-2] = -1
    for i, (z, model) in enumerate(models.items()):
        boost_case, boost_proj = model.predict([x, x_proj])
        boost = boost_case * boost_proj_ee2[i] / boost_proj
        boosts.append(boost)
    return boosts