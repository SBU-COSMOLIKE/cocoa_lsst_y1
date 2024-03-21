import numpy as np
import GPy
import logging
logger = logging.getLogger("GP")
logger.setLevel(logging.WARNING)

def get_redshifts(z_ini, output_redshifts, timestep_nsteps):
    redshift_endpoints = [z_ini] + output_redshifts
    scale_endpoints = [1/(1+z) for z in redshift_endpoints]
    das = [(scale_endpoints[i+1] - scale_endpoints[i])/timestep_nsteps[i] for i in range(len(timestep_nsteps))]
    scales = []
    for i in range(len(timestep_nsteps)):
        for j in range(timestep_nsteps[i]):
            scale = scale_endpoints[i] + j*das[i]
            scales.append(scale)
    zs_ = [round((1/a - 1.0), 3) for a in scales]
    zs_.append(0.0)
    zs_= np.flip(zs_)
    return zs_

param_dim = 5
num_points = 400
num_points_test = 10
redshifts = get_redshifts(20, [3,2,1,0.5,0], [12,5,8,9,17])
redshifts_ee2 = [redshifts[i] for i in range(40)]
z_cut = 34
n_bins_low_z = 512
n_bins_high_z = 256
n_pts_filter = 70
N_pc = 12
param_mins = [0.232, 0.039, 0.916, 1.66e-9, 0.604]
param_maxs = [0.408, 0.061, 1.004, 2.54e-9, 0.736]

def normalize_param(param_min, param_max, param):
    normalized_param = (param - param_min)/(param_max - param_min)
    return normalized_param
def unnormalize_param(param_min, param_max, normalized_param):
    param = (normalized_param * (param_max - param_min)) + param_min
    return param
def initialize_emulator(all_hyperparams,qs_reduced_,lhs):
    all_gps = []
    for z_index in range(len(redshifts_ee2)):
        all_gps.append([])
        for pc_index in range(N_pc):
            x = []
            y=[]
            for i in range(len(qs_reduced_[z_index])):
                x.append(lhs[i])
                q_of_k = qs_reduced_[z_index][i][pc_index]
                y.append([q_of_k])
            kernel = GPy.kern.RBF(input_dim=param_dim, ARD = True)
            x=np.array(x)
            y=np.array(y)
            m = GPy.models.GPRegression(x,y,kernel)
            m.rbf.variance[0] = all_hyperparams[z_index][pc_index][0]
            m.Gaussian_noise.variance[0] = all_hyperparams[z_index][pc_index][1]
            m.rbf.lengthscale[0:] = all_hyperparams[z_index][pc_index][2:]
            all_gps[z_index].append(m)
    return all_gps
def inv_pc(pcs_,mean_,vec):
    expanded_q = [mean_[i] for i in range(len(mean_))]
    for pc_index in range(len(pcs_)):
        expanded_q += vec[pc_index]*pcs_[pc_index]
    return expanded_q

def emulate_all_zs(params_, all_gps, qs_reduced, all_pcs_, all_means_, ks_in, zs_in):
    emulated_qs = []
    emulation_uncertainties = []
    for z_index in range(len(redshifts_ee2)):
        emulated_reduced_q = []
        emulation_uncertainty2 = 0
        for pc_index in range(N_pc):
            params_to_predict = np.array([params_])
            m = all_gps[z_index][pc_index]
            pred, pred_var = m.predict(params_to_predict)
            emulated_reduced_q.append(pred[0][0])
            the_mean = [entry for entry in all_means_[z_index]]
        emulated_q = inv_pc(all_pcs_[z_index], the_mean, emulated_reduced_q)
        emulated_qs.append(emulated_q)
    return emulated_qs

def find_crossing_index(arr, value):
    index = np.searchsorted(arr, value, side='left')
    return index
