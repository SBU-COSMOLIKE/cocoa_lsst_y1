from numpy import loadtxt, array, clip
import logging
import euclidemu2 as ee2
from dataclasses import dataclass
from scipy.interpolate import interp1d
from pathlib import Path
from re import search, findall
from pandas import DataFrame, Series
from .config import *
import camb


@dataclass
class CosmoParams:
    h: float
    ob: float
    om: float
    As: float
    ns: float
    w: float
    wa: float
        
def format_z(z):
    return f"{float(z):.3f}"

def get_ee2(cosmo_params, target_ks, z_val=0.000, lcdm=False):
    cosmo_par={'As':cosmo_params[3],
               'Omm':cosmo_params[2],
               'Omb':cosmo_params[1],
               'h':cosmo_params[0],
               'ns':cosmo_params[4],
               'mnu':0.058,
               'w' :-1.0 if lcdm else cosmo_params[5],
               'wa':0.0 if lcdm else cosmo_params[6]}

    ks, b = ee2.get_boost(cosmo_par,z_val, target_ks)
    if isinstance(z_val, (int, float)):
        return b[0]
    else:
        return b


def load_cola(path, z, use_ref=False):
    logging.info(f"Loading simulation data at z={z} from path {path}")
    lhs = loadtxt(f"{path}/lhs.txt")
    num_samples = len(lhs)
    
    z_idx = zs_cola.tolist().index(float(z))
    k_max = int(k_maxs[z_idx]) #Accounts for header
    
    pks_lin = []
    pks_nl  = []

    for i in range(num_samples):
        ks, pk_nl_a, pk_lin = loadtxt(f"{path}/sims/a/z{format_z(z)}/pofk_run_{i}_cb_z{format_z(z)}.txt",
                                      unpack=True, usecols=(0,1,2), skiprows=2, max_rows=k_max)
        ks, pk_nl_b, pk_lin = loadtxt(f"{path}/sims/b/z{format_z(z)}/pofk_run_{i}_cb_z{format_z(z)}.txt", 
                                      unpack=True, usecols=(0,1,2), skiprows=2, max_rows=k_max)
        pk_nl = 0.5*(pk_nl_a + pk_nl_b)
        if any(pk_nl < shot_noise_pk):
            print(f"Warning: power spectrum for cosmology {i} at z = {z} is less than shot noise")
        else: pk_nl -= shot_noise_pk
        pks_lin.append(pk_lin)
        pks_nl.append(pk_nl)
    
    if use_ref:
        logging.info(f"Collecting data from reference cosmology")
        ks, pk_nl_a_ref, pk_lin_ref = loadtxt(f"{path}/sims/a/z{format_z(z)}/pofk_run_{REF_IDX}_cb_z{format_z(z)}.txt",
                                              unpack=True, usecols=(0,1,2), skiprows=2,
                                              max_rows=k_max)
        ks, pk_nl_b_ref, pk_lin_ref = loadtxt(f"{path}/sims/b/z{format_z(z)}/pofk_run_{REF_IDX}_cb_z{format_z(z)}.txt",
                                              unpack=True, usecols=(0,1,2), skiprows=2, 
                                              max_rows=k_max)
        pk_nl_ref = 0.5*(pk_nl_a_ref + pk_nl_b_ref)
        pk_nl_ref -= shot_noise_pk
        pks_lin.append(pk_lin_ref)
        pks_nl.append(pk_nl_ref)
        
    pks_lin = array(pks_lin)
    pks_nl  = array(pks_nl)
    
    logging.info(f"Simulation data loaded")
    return lhs, ks, pks_lin, pks_nl


def load_EE2_df(input_type, z, target_ks=k_vals, use_ref=False):
    """
    Load EE2 boost factors at a given redshift and interpolate to the target k values.
    
    Parameters:
    - path: Path to the EE2 directory
    - z: Redshift (should match the filenames, e.g. z=0.000)
    - target_ks: 1D array of k values to interpolate to
    
    Returns:
    - boosts_df: Pandas DataFrame with columns ['k', 'EE2 1', 'EE2 4', ...]
    """    
    z_idx = zs_cola.tolist().index(float(z))
    k_max = int(k_maxs[z_idx])
    target_ks = k_vals[:k_max]
    
    ee2_dir = Path(f"./{input_type}/EE2/z{format_z(z)}/")
    
    logging.info(f"Loading EE2 boost files at z={z} from path {ee2_dir}")
    file_list = sorted([f for f in ee2_dir.glob(f"boost_*_z{format_z(z)}.txt") if f.is_file()])
    
    if not file_list:
        raise FileNotFoundError(f"No EE2 boost files found at redshift z={z} in {ee2_dir}")
    
    boost_data = {"k": target_ks}
    
    for file in file_list:
        # Extract sim number from filename using regex
        match = search(r"boost_(\d+)_z", file.name)
        if not match:
            logging.warning(f"Could not parse simulation number from filename: {file.name}")
            continue
        sim_idx = int(match.group(1))
        
        if sim_idx == REF_IDX and not use_ref:
            continue
        
        data = loadtxt(file, skiprows=1)
        k_vals, boost_vals = data[:, 0], data[:, 1]
        
        interp = interp1d(k_vals, boost_vals, fill_value='extrapolate')
        
        name = "EE2 Ref" if sim_idx == REF_IDX else f"EE2 {sim_idx}"
        boost_data[name] = interp(target_ks)

    boosts_df = DataFrame(boost_data)
    boosts_df.index = target_ks
    boosts_df.index.name = "k"
    logging.info(f"Loaded and interpolated {len(boost_data) - 1} EE2 boost files")  # -1 for 'k' column
    
    return boosts_df


def lcdm_projection(params):
    """
    First confines the params to the EE2 box then projects onto LCDM

    Returns:
        np.array: LCDM and EE2 projected parameters
    """  

    proj_params = clip(params, ee2_mins, ee2_maxs)
    proj_params = CosmoParams(*proj_params)
    
    proj_params.w = -1.0
    proj_params.wa = 0.0

    return array(list(vars(proj_params).values()))


def get_cosmologies(df):
    """
    Takes dataframe of boosts with cosmo numbers in the column names and provides
    a list with all the cosmo numbers in that dataframe. Useful for finding the cosmos
    in the training set that are also in the EE2 box

    Returns:
        List: LCDM and EE2 projected parameters
    """  
    pattern = r'\b(\d+)\b'  # regex pattern to match a space followed by one or more digits,
                            # ensuring it's followed by a non-digit character or end of string

    matches = [int(match) for col in df.columns for match in findall(pattern, col)]
    return list(set(matches))


def get_ee2_cosmos(df):
    # Start with all rows marked as valid (True)
    mask = Series([True] * len(df))

    # Loop through each parameter by index
    for i, param in enumerate(PARAM_COLS):
        mask &= (df[param] >= ee2_mins[i]) & (df[param] <= ee2_maxs[i])

    # Get the indices of rows where all parameters are within bounds
    matching_indices = df.index[mask].tolist()
    
    return matching_indices


def get_sigma8(params, redshifts=[0], tau = 0.078):
    # Note: Ref cosmology doesn't provide tau, using same as Guilherme
    k_min = 1e-2
    k_max = 3.14159
    n_points = 200
    params = CosmoParams(*params)
    cosmology = camb.set_params(# Background
                                    H0 = 100*params.h, 
                                    ombh2=params.ob*params.h**2, 
                                    omch2=params.om*params.h**2,
                                    TCMB = 2.7255,
                                    # Dark Energy
                                    dark_energy_model='ppf', w = params.w, 
                                    wa = params.wa,
                                    # Neutrinos
                                    nnu=3.046, mnu = 0.058, num_nu_massless = 0, num_massive_neutrinos = 3,
                                    # Initial Power Spectrum
                                    As = params.As, 
                                    ns = params.ns, tau = 0.0543,
                                    YHe = 0.246, WantTransfer=True)
    cosmology.set_matter_power(redshifts=redshifts, kmax=k_max)
    results = camb.get_results(cosmology)
    sigma8s = results.get_sigma8()
    return sigma8s
