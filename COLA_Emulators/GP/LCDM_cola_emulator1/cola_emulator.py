import numpy as np
import scipy
import scipy.stats
import scipy.interpolate as interpolate
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import CubicSpline
from scipy.fftpack import dst, idst
import scipy.integrate
import math
import sys, platform, os
import GPy

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
k_min = 1e-2
k_max = 3
n_k_bins = 200
num_points = 400
num_points_test = 10
redshifts = get_redshifts(20, [3,2,1,0.5,0], [12,5,8,9,17])
redshifts_ee2 = [redshifts[i] for i in range(40)]
N_pc = 11
param_mins = [0.232, 0.039, 0.916, 1.66e-9, 0.604]
param_maxs = [0.408, 0.061, 1.004, 2.54e-9, 0.736]

from scipy.fftpack import dst, idst
from scipy.integrate import simps

def smooth_bao(ks, pk):
    
    n = 10
    dst_ks = np.linspace(1e-4, 5, 2**n) #10
    logks = np.log(dst_ks)
    
    spline_loglog_pk_2 = interpolate.interp1d(np.log(ks), np.log(pk), kind='linear', fill_value='extrapolate')
    spline_loglog_pk2 = spline_loglog_pk_2(np.log(np.linspace(1e-4, 5, 2**n)))
    
    spline_loglog_pk = interpolate.splrep(np.log(np.linspace(1e-4,5, 2**n)), spline_loglog_pk2, s=0)    
        
    logkpk = np.log10(dst_ks * np.exp(interpolate.splev(np.log(dst_ks), spline_loglog_pk, der=0, ext=0)))
    sine_transf_logkpk = dst(logkpk, type=2)# dst(logkpk, type=2, norm='ortho')

    odds = [] # odd entries
    evens = [] # even entries
    even_is = [] # odd indices
    odd_is = [] # even indices
    all_is = [] # all indices
    for i, entry in enumerate(sine_transf_logkpk):
        all_is.append(i)
        if i%2 == 0:
            even_is.append(i)
            evens.append(entry)
        else:
            odd_is.append(i)
            odds.append(entry)
    odd_is=np.array(odd_is)
    even_is=np.array(even_is)
    odds=np.array(odds)
    evens=np.array(evens)

    odd_is = np.array(odd_is)
    even_is = np.array(even_is)
    
    odds_interp = interpolate.splrep(odd_is, odds, s=0) 
    evens_interp = interpolate.splrep(even_is, evens, s=0) 
    
    d2_odds =interpolate.splev(odd_is, odds_interp, der=2, ext=0)    
    d2_evens =interpolate.splev(even_is, evens_interp, der=2, ext=0)
    
    d2_odds_1 =interpolate.splev(odd_is +2, odds_interp, der=2, ext=0) 
    d2_evens_1 =interpolate.splev(even_is +2, evens_interp, der=2, ext=0)

    d2_odds_2 =interpolate.splev(odd_is - 2, odds_interp, der=2, ext=0)
    d2_evens_2 =interpolate.splev(even_is - 2 , evens_interp, der=2, ext=0)   
    
    d2_odds_avg = (d2_odds + d2_odds_2 + d2_odds_1)/3
  
    d2_evens_avg = (d2_evens + d2_evens_2 + d2_evens_1)/3 
    
    imin_even = 50+np.argmax(d2_evens_avg[50:150]) -9
    
    imax_even = 50+np.argmin(d2_evens_avg[50:150])+36

    imin_odd = 50+np.argmax(d2_odds_avg[50:150])-9

    imax_odd = 50+np.argmin(d2_odds_avg[50:150])+37       
    
    even_is_removed_bumps = np.concatenate((even_is[:imin_even], even_is[imax_even:]))
    odd_is_removed_bumps = np.concatenate((odd_is[:imin_odd], odd_is[imax_odd:]))

    evens_removed_bumps = np.concatenate((evens[:imin_even], evens[imax_even:]))
    odds_removed_bumps = np.concatenate((odds[:imin_odd], odds[imax_odd:]))

    even_holed_cs = interpolate.splrep(even_is_removed_bumps, evens_removed_bumps * (even_is_removed_bumps+1)**2, s=0)
    odd_holed_cs = interpolate.splrep(odd_is_removed_bumps, odds_removed_bumps * (odd_is_removed_bumps+1)**2, s=0)
   
    evens_treated = interpolate.splev(even_is, even_holed_cs, der=0, ext=0) / (even_is + 1)**2
    odds_treated = interpolate.splev(odd_is, odd_holed_cs, der=0, ext=0) / (odd_is + 1)**2
    treated_transform = []
    for odd, even in zip(odds_treated, evens_treated):
        treated_transform.append(even)
        treated_transform.append(odd)
    treated_transform=np.array(treated_transform)    
    treated_logkpk =idst(treated_transform, type=2)/ (2 * len(treated_transform)) # idst(treated_transform, type=2, norm='ortho')
    
    pk_nw = 10**(treated_logkpk)/dst_ks
    
    k_highk = ks[ks > 4]
    p_highk = pk[ks > 4]

    k_extended = np.concatenate((dst_ks[dst_ks < 4], k_highk))
    
    p_extended = np.concatenate((pk_nw[dst_ks < 4], p_highk))
    
    pksmooth_cs = interpolate.splrep(np.log(k_extended), np.log(p_extended), s=0)
    pksmooth_interp = np.exp(interpolate.splev(np.log(ks), pksmooth_cs, der=0, ext=0))       
   
    return pksmooth_interp#, d2_odds_avg, d2_evens_avg, odd_is,even_is, imin_odd,imax_odd, imin_even, imax_even, np.exp(interpolate.splev(np.log(dst_ks),spline_loglog_pk, der=0, ext=0)), dst_ks,np.argmax(d2_evens_avg[100:300]),np.argmin(d2_evens_avg[100:300]),np.argmax(d2_odds_avg[100:300]),np.argmin(d2_odds_avg[100:300]),spline_loglog_pk2

def smear_bao(ks, pk, pk_nw):
    
    from scipy.integrate import trapz  
    integral = simps(pk,ks)#trapz(ks * pk, x=np.log(ks))  #simps(pk,ks)#
    k_star_inv = (1.0/(3.0 * np.pi**2)) * integral
    Gk = np.array([np.exp(-0.5*k_star_inv * (k_**2)) for k_ in ks])
    pk_smeared = pk*Gk + pk_nw*(1.0 - Gk)
    return pk_smeared 
    
def turn_qk_to_b(ks, pk_l, qk):
    min_found = False
    max_found = False
    for i in range(len(ks)):
        if ks[i] >= 0.01:
            kmin_index = i
            min_found = True
            break
    if min_found == False:
        print('Error: Did not provide any k > 0.01')
    for i in range(len(ks)):
        if ks[len(ks) - i] <= 1.0:
            kmax_index = len(ks) - i
            max_found = True
            break
    if max_found == False:
        print('Error: Did not provide any k < 1.0')
    ks_smear = ks[kmin_index:kmax_index]
    pk_l_ = np.copy(pk_l[kmin_index:kmax_index])
    qk_ = np.copy(qk[kmin_index:kmax_index])
    pk_nw_ = smooth_bao(ks_smear, pk_l_)
    pk_smeared_ = smear_bao(ks_smear, pk_l_, pk_nw_)
    pk_smeared = np.concatenate([pk_l[0:kmin_index],pk_smeared_,pk_l[kmax_index:len(ks)]])
    b = (pk_smeared * np.exp(qk))/pk_l
    return b

def normalize_param(param_min, param_max, param):
    normalized_param = (param - param_min)/(param_max - param_min)
    #normalized_param = param
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
def emulate_all_zs_old(params_, all_gps, qs_reduced, all_pcs_, all_means_, ks_in, ks_out, zs_in, zs_out):
    emulated_qs_ = []
    emulation_uncertainties = []
    for z_index in range(len(redshifts_ee2)):
        emulated_reduced_q = []
        emulation_uncertainty2 = 0
        for pc_index in range(N_pc):
            params_to_predict = np.array([params_])
            m = all_gps[z_index][pc_index]
            pred, pred_var = m.predict(params_to_predict)
            emulated_reduced_q.append(pred[0][0])
            emulation_uncertainty2 = emulation_uncertainty2 + pred_var[0]**2
            the_mean = [entry for entry in all_means_[z_index]]
        emulated_q = inv_pc(all_pcs_[z_index], the_mean,emulated_reduced_q)
        #emulated_b = turn_qk_to_b(params_, z_index, emulated_q)
        emulated_qs_.append(emulated_q)
        emulation_uncertainty = math.sqrt(emulation_uncertainty2)
        emulation_uncertainties.append(emulation_uncertainty)
    emulated_qs_interp = interp2d(ks_in,zs_in,emulated_qs_)
    emulated_qs = emulated_qs_interp(ks_out, zs_out)
    return emulated_qs, emulation_uncertainties

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
        #emulated_b = turn_qk_to_b(params_, z_index, emulated_q)
        emulated_qs.append(emulated_q)
    return emulated_qs

def find_crossing_index(arr, value):
    index = np.searchsorted(arr, value, side='left')
    return index
