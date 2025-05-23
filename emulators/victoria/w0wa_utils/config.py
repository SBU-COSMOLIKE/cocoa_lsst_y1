from numpy import loadtxt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# base_path = "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/./projects/lsst_y1/emulators/victoria/"
import os
cocoa_path = os.getcwd()
base_path = cocoa_path + "/projects/lsst_y1/emulators/victoria/"

shot_noise_pk = 1

NUM_PCS = 15

# zs_cola = [
#     0.000, 0.020, 0.041, 0.062, 0.085, 0.109, 0.133, 0.159, 0.186, 0.214, 0.244, 0.275, 0.308, 
#     0.342, 0.378, 0.417, 0.457, 0.500, 0.543, 0.588, 0.636, 0.688, 0.742, 0.800, 0.862, 0.929, 
#     1.000, 1.087, 1.182, 1.286, 1.400, 1.526, 1.667, 1.824, 2.000, 2.158, 2.333, 2.529, 2.750, 
#     3.000
# ]

zs_cola, k_maxs = loadtxt(base_path+"kmax_vals.txt", unpack=True, usecols=(0,1), delimiter=',')

k_vals = loadtxt(base_path+"ks.txt", skiprows=2)

PARAM_DIR = base_path + 'params_files'
BOOST_DIR = base_path + 'boost_files'
IMGS_DIR = base_path + 'imgs'
MODELS_DIR = base_path + 'models'

PARAM_COLS = ["h", "Omegab", "Omegam", "As", "ns", "w", "wa"]

ee2_mins = [0.61, 0.04, 0.24, 1.7e-9, 0.92, -1.3, -0.7]
ee2_maxs = [0.73, 0.06, 0.40, 2.5e-9, 1.00, -0.7, 0.5]
    
REF_IDX = 9999

VER = 4