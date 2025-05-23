import pickle
import keras
import numpy as np
import euclidemu2 as ee2
from copy import copy
from scipy.interpolate import interp1d
from keras.models import load_model
from joblib import load
from w0wa_utils.config import VER
from w0wa_utils.model import CustomActivationLayer


import os
emulator_dir = os.path.dirname(os.path.abspath(__file__))

zs_cola, k_maxs = np.loadtxt(f"{emulator_dir}/kmax_vals.txt", unpack=True, usecols=(0,1), delimiter=',')
ks = np.loadtxt(f"{emulator_dir}/ks.txt", skiprows=2)
log10ks = np.log10(ks)

pcs=15
ver=3

def format_z(z):
    return f"{float(z):.3f}"


print("[colaemu] Loading models")
models = {}
boost_scalers = {}
pcas = {}
param_scaler = load(f"{emulator_dir}/params_files/scaler_v{VER}_z{format_z(0.0)}.pkl")

for z in zs_cola:
    models[z] = load_model(f"{emulator_dir}/models/NN_{pcs}_v{VER}_z{format_z(z)}.keras") 
    boost_scalers[z] = load(f"{emulator_dir}/boost_files/scaler_v{VER}_z{format_z(z)}.pkl")
    pcas[z] = load(f"{emulator_dir}/boost_files/pca_v{VER}_z{format_z(z)}.pkl")
print("[colaemu] Models loaded")

def predict_logboost(x, z):
    x_norm = param_scaler.transform([x])
    pcs = models[z]([x_norm])
    logboost_norm = pcas[z].inverse_transform(pcs)
    logboost = boost_scalers[z].inverse_transform(logboost_norm)

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
        if VER==3:
            k_max = 255 
        else:
            k_max = int(k_maxs[i])
        x_norm = param_scaler.transform([x, x_proj])
        pcs = models[z](x_norm)
        logboost_norm = pcas[z].inverse_transform(pcs)
        boost_case, boost_proj = np.exp(boost_scalers[z].inverse_transform(logboost_norm))
        if k_custom is None: boost = boost_case * boost_proj_ee2[i][:k_max] / boost_proj
        else:
            log10ks_cutoff = log10ks[:k_max]
            ratio = boost_case/boost_proj
            interp = interp1d(
                log10ks_cutoff,
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
