import sys, platform, os
sys.path.insert(0, os.environ['ROOTDIR'] + 
                   '/external_modules/code/CAMB/build/lib.linux-x86_64-'
                   +os.environ['PYTHON_VERSION'])
import functools
import numpy as np
import ipyparallel
import sys, platform, os
import math
import euclidemu2
import scipy
from getdist import IniFile
import itertools
import iminuit
from mpi4py.futures import MPIPoolExecutor
import camb
import cosmolike_lsst_y1_interface as ci

def get_camb_cosmology(omegam, omegab, H0, ns, As_1e9 , w, w0pwa, mnu,
                       AccuracyBoost=1.0, kmax=10, k_per_logint=20, 
                       CAMBAccuracyBoost=1.1, non_linear_emul=2):
    As = lambda As_1e9: 1e-9 * As_1e9
    wa = lambda w0pwa, w: w0pwa - w
    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
    omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

    CAMBAccuracyBoost = CAMBAccuracyBoost*AccuracyBoost
    kmax = max(kmax/2.0, kmax*(1.0 + 3*(AccuracyBoost-1)))
    k_per_logint = max(k_per_logint/2.0, int(k_per_logint) + int(3*(AccuracyBoost-1)))
    extrap_kmax = max(max(2.5e2, 3*kmax), max(2.5e2, 3*kmax) * AccuracyBoost)

    z_interp_1D = np.concatenate( (np.concatenate((np.linspace(0,2.0,1000),
                                                   np.linspace(2.0,10.1,200)),
                                                   axis=0),
                                   np.linspace(1080,2000,20)),
                                   axis=0)
    
    z_interp_2D = np.concatenate(( np.linspace(0, 2.0, 95), 
                                   np.linspace(2.25, 10, 5)),  
                                 axis=0)

    log10k_interp_2D = np.linspace(-4.2, 2.0, 1200)

    pars = camb.set_params(H0=H0, 
                           ombh2=omegabh2(omegab, H0), 
                           omch2=omegach2(omegam, omegab, mnu, H0), 
                           mnu=mnu, 
                           omk=0, 
                           tau=0.06,  
                           As=As(As_1e9), 
                           ns=ns, 
                           halofit_version='takahashi', 
                           lmax=10,
                           AccuracyBoost=CAMBAccuracyBoost,
                           lens_potential_accuracy=1.0,
                           num_massive_neutrinos=1,
                           nnu=3.046,
                           accurate_massive_neutrino_transfers=False,
                           k_per_logint=k_per_logint,
                           kmax = kmax);
    
    pars.set_dark_energy(w=w, wa=wa(w0pwa, w), dark_energy_model='ppf');    
    
    pars.NonLinear = camb.model.NonLinear_both
    
    pars.set_matter_power(redshifts = z_interp_2D, kmax = kmax, silent = True);
    results = camb.get_results(pars)
    
    PKL  = results.get_matter_power_interpolator(var1="delta_tot", 
                                                 var2="delta_tot", 
                                                 nonlinear=False, 
                                                 extrap_kmax=extrap_kmax, 
                                                 hubble_units=False, 
                                                 k_hunit=False);
    
    PKNL = results.get_matter_power_interpolator(var1="delta_tot", 
                                                 var2="delta_tot",  
                                                 nonlinear=True, 
                                                 extrap_kmax=extrap_kmax, 
                                                 hubble_units=False, 
                                                 k_hunit=False);
    
    lnPL = np.empty(len(log10k_interp_2D)*len(z_interp_2D))
    for i in range(len(z_interp_2D)):
        lnPL[i::len(z_interp_2D)] = np.log(PKL.P(z_interp_2D[i], 
                                                 np.power(10.0,log10k_interp_2D)))
    lnPL  += np.log(((H0/100.)**3)) 
    
    lnPNL  = np.empty(len(log10k_interp_2D)*len(z_interp_2D))
    if non_linear_emul == 1:
        params = { 'Omm'  : omegam, 
                   'As'   : As(As_1e9), 
                   'Omb'  : omegab,
                   'ns'   : ns, 
                   'h'    : H0/100., 
                   'mnu'  : mnu,  
                   'w'    : w, 
                   'wa'   : wa(w0pwa, w)
                 }
        kbt, bt = euclidemu2.get_boost( params, 
                                        z_interp_2D, 
                                        np.power(10.0, np.linspace(-2.0589, 0.973, len(log10k_interp_2D)))
                                      )
        log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
        
        for i in range(len(z_interp_2D)):    
            lnbt = scipy.interpolate.interp1d(np.log10(kbt), 
                                              np.log(bt[i]), 
                                              kind = 'linear', 
                                              fill_value = 'extrapolate', 
                                              assume_sorted = True)(log10k_interp_2D)
            lnbt[np.power(10,log10k_interp_2D) < 8.73e-3] = 0.0
            lnPNL[i::len(z_interp_2D)] = lnPL[i::len(z_interp_2D)] + lnbt
    elif non_linear_emul == 2:
        for i in range(len(z_interp_2D)):
            lnPNL[i::len(z_interp_2D)] = np.log(PKNL.P(z_interp_2D[i], 
                                                       np.power(10.0,log10k_interp_2D)))            
        log10k_interp_2D = log10k_interp_2D - np.log10(H0/100.)
        lnPNL += np.log(((H0/100.)**3))

    G_growth = np.sqrt(PKL.P(z_interp_2D,0.0005)/PKL.P(0,0.0005))
    G_growth = G_growth*(1 + z_interp_2D)/G_growth[len(G_growth)-1]

    chi = results.comoving_radial_distance(z_interp_1D, tol=1e-4) * (H0/100.)

    return (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, G_growth, z_interp_1D, chi)

def chi2(omegam=0.3, As_1e9=2.1, ns=0.96605, H0=67, 
         w0pwa=-0.9, mnu=0.06, w=-0.9, omegab=0.04, 
         LSST_DZ_S1=0.0414632, LSST_DZ_S2=0.00147332,
         LSST_DZ_S3=0.0237035, LSST_DZ_S4=-0.0773436,
         LSST_DZ_S5=-8.67127e-05, LSST_M1=0.0191832,
         LSST_M2=-0.0431752, LSST_M3=-0.034961,
         LSST_M4=-0.0158096, LSST_M5 = -0.0158096,
         LSST_A1_1=0.606102, LSST_A1_2=-1.51541,
         AccuracyBoost=1.0, non_linear_emul=2):

    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL, 
        G_growth, z_interp_1D, chi) = get_camb_cosmology(omegam=omegam, 
                                                         omegab=omegab, 
                                                         H0=H0, 
                                                         ns=ns, 
                                                         As_1e9=As_1e9,
                                                         w=w, 
                                                         w0pwa=w0pwa, 
                                                         mnu=mnu,
                                                         AccuracyBoost=AccuracyBoost,
                                                         non_linear_emul=non_linear_emul)
    
    ci.init_accuracy_boost(AccuracyBoost, 
                           AccuracyBoost, 
                           int(1+5*(AccuracyBoost-1)))
    
    ci.set_cosmology(omegam=omegam,
                     H0=H0, 
                     log10k_2D=log10k_interp_2D, 
                     z_2D=z_interp_2D, 
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_1D=z_interp_1D,
                     chi=chi)
    
    ci.set_nuisance_shear_calib(M = [LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5])
    
    ci.set_nuisance_shear_photoz(bias = [LSST_DZ_S1, LSST_DZ_S2, LSST_DZ_S3, LSST_DZ_S4, LSST_DZ_S5])
    
    ci.set_nuisance_ia(A1 = [LSST_A1_1, LSST_A1_2, 0, 0, 0], 
                       A2 = [0, 0, 0, 0, 0], 
                       B_TA = [0, 0, 0, 0, 0])

    datavector = np.array(ci.compute_data_vector_masked())
    
    return ci.compute_chi2(datavector)

def foo_ns(params, *args):
        omegam = params[0]
        H0     = params[1]
        omegab = params[2]
        As_1e9 = params[3]
        ns, AccuracyBoost, non_linear_emul = args
        return chi2(omegam=omegam, 
                    H0=H0, 
                    omegab=omegab,
                    As_1e9=As_1e9,
                    ns=ns,
                    AccuracyBoost=AccuracyBoost,
                    non_linear_emul=non_linear_emul)

def min_chi2(ns, func, x0, bounds, min_method, AccuracyBoost=1.0, 
             tol=0.01, maxfev=300000, non_linear_emul=2):

    args = (ns, AccuracyBoost, non_linear_emul)
    
    if min_method == 1:
        tmp = scipy.optimize.basinhopping(func, 
                                          x0, 
                                          T=0.45, 
                                          target_accept_rate=0.3, 
                                          niter=10, 
                                          stepsize=0.1,
                                          minimizer_kwargs={"method": 'Nelder-Mead', 
                                                            "args": args, 
                                                            "bounds": bounds, 
                                                            "options": {'adaptive' : True, 
                                                                        'fatol' : tol, 
                                                                        'maxfev' : maxfev}})
    elif min_method == 2:
        tmp = iminuit.minimize(func, 
                               x0, 
                               args=args, 
                               bounds=bounds, 
                               method="migrad", 
                               tol=tol,
                               options = {'stra' : 1, 'maxfun': maxfev})
    elif min_method == 3:
        tmp = scipy.optimize.minimize(func, 
                                      x0, 
                                      args=args, 
                                      method='Nelder-Mead', 
                                      bounds=bounds, 
                                      options = {'adaptive' : True, 
                                                 'fatol' : tol, 
                                                 'maxfev' : maxfev})
    elif min_method == 4:
        tmp = scipy.optimize.minimize(func, 
                                      x0, 
                                      args=args, 
                                      method='Powell', 
                                      bounds = bounds, 
                                      options = {'xtol' : tol/2, 
                                                 'ftol' : tol, 
                                                 'maxfev' : maxfev})
    elif min_method == 5:
        # https://stats.stackexchange.com/a/456073
        tmp = scipy.optimize.dual_annealing(func=func, 
                                            x0=x0, 
                                            args=args, 
                                            bounds=bounds, 
                                            maxfun=maxfev,
                                            no_local_search=True, 
                                            maxiter=10, 
                                            visit=1.01, 
                                            accept=1, 
                                            initial_temp=5230.0, 
                                            restart_temp_ratio=0.0002)    
    return tmp.fun

# TO RUN THIS SCRIPT
# mpirun -n 12 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} python -m mpi4py.futures EXAMPLE_PROFILE1.py

if __name__ == '__main__':
    # profile likelihood on ns
    ns = np.arange(0.90, 1.02, 0.01)
    x0 = [0.35, 70, 0.04, 2.12] 
    bounds = [[0.2,0.4], [50, 90], [0.03, 0.07], [2.00, 2.35]]
    non_linear_emul=2
    AccuracyBoost=1.0
    tol=0.01
    maxfev=200000

    min_method = 4

    CLprobe="xi"
    path= os.environ['ROOTDIR'] + "external_modules/data/lsst_y1"
    data_file="LSST_Y1_M1_GGL0.05.dataset"

    IA_model = 0
    IA_redshift_evolution = 3

    # Init Cosmolike & Read LSST-Y1 data file
    ini = IniFile(os.path.normpath(os.path.join(path, data_file)))
    data_vector_file = ini.relativeFileName('data_file')
    cov_file = ini.relativeFileName('cov_file')
    mask_file = ini.relativeFileName('mask_file')
    ntheta = ini.int("n_theta")
    theta_min_arcmin = ini.float("theta_min_arcmin")
    theta_max_arcmin = ini.float("theta_max_arcmin")

    lens_file = ini.relativeFileName('nz_lens_file')

    source_file = ini.relativeFileName('nz_source_file')

    lens_ntomo = ini.int("lens_ntomo")

    source_ntomo = ini.int("source_ntomo")

    ci.initial_setup()

    ci.init_cosmo_runmode(is_linear = False)

    ci.init_source_sample( filename = source_file, 
                           ntomo_bins = int(source_ntomo))

    ci.init_lens_sample( filename=lens_file, 
                         ntomo_bins=int(lens_ntomo))

    ci.init_IA( ia_model = int(IA_model), 
                ia_redshift_evolution = int(IA_redshift_evolution))

    # Init Cosmolike
    ci.init_probes(possible_probes = CLprobe)
    ci.init_binning(int(ntheta), theta_min_arcmin, theta_max_arcmin)
    ci.init_data_real(cov_file, mask_file, data_vector_file)

    print("OK")

    #print(foo_ns([0.3, 67.32, 0.04, 2.1], 0.96605, AccuracyBoost, non_linear_emul))
    #executor = MPIPoolExecutor()
    #result = np.array(list(executor.map(functools.partial(min_chi2, 
    #                                                      func=foo_ns,
    #                                                      x0=x0, 
    #                                                      bounds=bounds,
    #                                                      AccuracyBoost=AccuracyBoost,
    #                                                      maxfev=maxfev,
    #                                                      non_linear_emul=non_linear_emul,
    #                                                      min_method=min_method, 
    #                                                      tol=tol), 
    #                                    ns)))
    #executor.shutdown()


    #np.savetxt("file1.txt", result)