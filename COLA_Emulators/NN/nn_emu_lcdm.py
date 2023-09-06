# NN COLA Emulator
import os
import multiprocessing
from functools import partial
import multiprocessing
import concurrent
import numpy as np
import keras

lims = {}
lims['h'] = [0.61, 0.73]
lims['Omegab'] = [0.04, 0.06]
lims['Omegam'] = [0.24, 0.4]
lims['As'] = [1.7e-9, 2.5e-9]
lims['ns'] = [0.92, 1]
lims['w'] = [-1.3, -0.7]

num_pcs = 11
path_to_emulator =  os.path.dirname(__file__)
cola_redshifts = np.loadtxt(f"{path_to_emulator}/data/cola_zs.txt")
cola_ks_default = np.loadtxt(f"{path_to_emulator}/data/cola_ks_default.txt")
cola_ks_high = None

num_cpus = multiprocessing.cpu_count()

def normalize_params(params):
    '''
    Takes a params = (h, Omegam, Omegab, As10to9, ns, wde) array and returns normalized parameters.
    Normalization is given by normalized_param = (param - param_min)(param_max - param_min).
    '''
    assert len(params) == 5, f"LCDM Emulator accepts 5 parameters, supplying {len(params)}."
    Omegam, Omegab, ns, As, h = params
    assert Omegam > lims['Omegam'][0] and Omegam < lims['Omegam'][1], f"Omega_matter = {Omegam} not in the valid range [{lims['Omegam'][0]}, {lims['Omegam'][1]}]"
    assert Omegab > lims['Omegab'][0] and Omegab < lims['Omegab'][1], f"Omega_baryon = {Omegab} not in the valid range [{lims['Omegab'][0]}, {lims['Omegab'][1]}]"
    assert ns > lims['ns'][0] and ns < lims['ns'][1], f"ns = {ns} not in the valid range [{lims['ns'][0]}, {lims['ns'][1]}]"
    assert As > lims['As'][0] and As < lims['As'][1], f"As = {As} not in the valid range [{lims['As'][0]}, {lims['As'][1]}]"
    assert h > lims['h'][0] and h < lims['h'][1], f"h = {h} not in the valid range [{lims['h'][0]}, {lims['h'][1]}]"
    
    normalized_params = [
        (Omegam-lims['Omegam'][0])/(lims['Omegam'][1] - lims['Omegam'][0]),
        (Omegab-lims['Omegab'][0])/(lims['Omegab'][1] - lims['Omegab'][0]),
        (ns-lims['ns'][0])/(lims['ns'][1] - lims['ns'][0]),
        (As-lims['As'][0])/(lims['As'][1] - lims['As'][0]), 
        (h-lims['h'][0])/(lims['h'][1] - lims['h'][0])
    ]

    return np.array(normalized_params)

def load_models():
    """
    Returns an array of NN models for default-precision LCDM for each COLA redshift z.
    """
    nn_models = []
    for i, z in enumerate(cola_redshifts):
        nn_models.append(keras.models.load_model(f"{path_to_emulator}/models/LCDM/NN_Z{z:.3f}"))
    return nn_models


def load_pcas():
    """
    Returns an array of PC basis and averages to inverse transform the NN outputs.
    For reference, the inverse transform of a set of components {PC_i} is:
        f = avg + \sum_i PC_i * basis_i
    """
    avgs = np.zeros((len(cola_redshifts), len(cola_ks_default)))
    pcs = np.zeros((len(cola_redshifts), 11, len(cola_ks_default)))
    for i, z in enumerate(cola_redshifts):
        avgs[i] = np.loadtxt(f"{path_to_emulator}/data/averages_lcdm/avg_{i}.txt")
        pcs[i] = np.loadtxt(f"{path_to_emulator}/data/pc_basis_lcdm/pcs_{i}.txt")
    return pcs, avgs
    
def load_normalization_factors():
    """
    Returns an array of normalization factors to rescale the Qs.
    """
    mins = np.loadtxt(f"{path_to_emulator}/data/mins_lcdm.txt")
    maxs = np.loadtxt(f"{path_to_emulator}/data/maxs_lcdm.txt")
    return mins, maxs

def inverse_transform(components, z_index):
    """
    Inverse transform PCA components into normalized Qs.
    """
    global pcs, avgs
    result = avgs[z_index]
    for i in range(num_pcs):
        result += components[i] * pcs[z_index, i]
    return result

def get_logboost_at_z(z_index, cosmo_params):
    As = cosmo_params['As']
    Omega_m = cosmo_params['Omm']
    Omega_b = cosmo_params['Omb']
    ns = cosmo_params['ns']
    h = cosmo_params['h']
    norm_params = normalize_params([Omega_m, Omega_b, ns, As, h])
        
    principal_components = nn_models[z_index](np.array([norm_params]))[0]
    norm_q = inverse_transform(principal_components, z_index)
    q = norm_q * (maxs[z_index] - mins[z_index]) + mins[z_index]
    return q

def get_logboost_parallel(cosmo_params):
    z_indices = list(range(len(cola_redshifts)))
    qs = np.empty((len(cola_redshifts), len(cola_ks_default)))
    get_logboost_at_z_for_cosmo = partial(get_logboost_at_z, cosmo_params=cosmo_params)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cpus) as executor:
        # Start the load operations and mark each future with its item
        gen = executor.map(get_logboost_at_z_for_cosmo, z_indices)
        for i, x in enumerate(gen):
            qs[i] = x
    return qs

def get_logboost(cosmo_params):
    qs = np.empty((len(cola_redshifts), len(cola_ks_default)))
    for i, z in enumerate(cola_redshifts):
        qs[i] = get_logboost_at_z(i, cosmo_params)
    return qs   

def get_boost(cosmo_params, ks = cola_ks_default, zs = cola_redshifts):
    """
    Returns an array of boosts at given ks and zs for the cosmology defined in cosmo_params
    Inputs:
        - `cosmo_params`: a dictionary of cosmological parameters. The keys to be defined are the same as EE2: `Omm`, `Omb`, `ns`, `As`, `h`, `w`.
        - `ks`: array of scales to return
        - `zs`: array of redshifts to return
    """
    return np.exp(get_logboost(cosmo_params))


mins, maxs = load_normalization_factors()
pcs, avgs = load_pcas()
nn_models = load_models()