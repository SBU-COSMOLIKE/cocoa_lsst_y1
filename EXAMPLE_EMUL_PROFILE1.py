import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered in subtract.*",
    module=r"emcee\.moves\.mh"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered in exp.*"
)
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from getdist import IniFile
from schwimmbad import MPIPool
import sys, platform, os
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_PROFILE1')
parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=5000)
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/lsst_y1/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="example_profile1")
parser.add_argument("--profile",
                    dest="profile",
                    help="Which Parameter to Profile",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1)
parser.add_argument("--factor",
                    dest="factor",
                    help="Factor that set the bounds (multiple of cov matrix)",
                    type=int,
                    nargs='?',
                    const=1,
                    default=3)
parser.add_argument("--numpts",
                    dest="numpts",
                    help="Number of Points to Compute Minimum",
                    type=int,
                    nargs='?',
                    const=1,
                    default=20)
parser.add_argument("--cov",
                    dest="cov",
                    help="Chain Covariance Matrix",
                    nargs='?',
                    const=1,
                    default="EXAMPLE_MCMC1.covmat")
parser.add_argument("--nwalkers",
                    dest="nwalkers",
                    help="Number of emcee walkers",
                    nargs='?',
                    const=1)
parser.add_argument("--minfile",
                    dest="minfile",
                    help="Minimization Result",
                    nargs='?',
                    const=1)
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
yaml_string=r"""
likelihood:
  lsst_y1.cosmic_shear:
    path: ./external_modules/data/lsst_y1
    data_file: lsst_y1_M1_GGL0.05.dataset   # 705 non-masked elements  (EE2 delta chi^2 ~ 11.8)
    use_emulator: True
    print_datavector: False
    print_datavector_file: "./projects/lsst_y1/chains/example1_lsst_y1_theory_emul.modelvector"
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.65
    proposal: 0.4
    latex: 10^9 A_\mathrm{s}
    drop: true
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 5
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
    drop: true
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.02
    proposal: 0.02
    latex: \Omega_\mathrm{m}
    drop: true
  mnu:
    value: 0.06
  omegabh2:
    value: 'lambda omegab, H0: omegab*(H0/100)**2'
    latex: \Omega_\mathrm{b} h^2
  omegach2:
    value: 'lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708'
    latex: \Omega_\mathrm{c} h^2
  logA:
    value: 'lambda As_1e9: np.log(10*As_1e9)'
  LSST_BARYON_Q1:
    value: 0.0
    latex: Q1_\mathrm{LSST}^1
  LSST_BARYON_Q2:
    value: 0.0
    latex: Q2_\mathrm{LSST}^2
  # WL photo-z errors
  LSST_DZ_S1:
    prior:
      dist: norm
      loc: 0.0414632
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0414632
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^1
  LSST_DZ_S2:
    prior:
      dist: norm
      loc: 0.00147332
      scale: 0.002
    ref:
      dist: norm
      loc: 0.00147332
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^2
  LSST_DZ_S3:
    prior:
      dist: norm
      loc: 0.0237035
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0237035
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^3
  LSST_DZ_S4:
    prior:
      dist: norm
      loc: -0.0773436
      scale: 0.002
    ref:
      dist: norm
      loc: -0.0773436
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^4
  LSST_DZ_S5:
    prior:
      dist: norm
      loc: -8.67127e-05
      scale: 0.002
    ref:
      dist: norm
      loc: -8.67127e-05
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,LSST}^5
  # Intrinsic alignment
  LSST_A1_1:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{1-IA,LSST}^1
  LSST_A1_2:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.5
  # Shear calibration parameters
  LSST_M1:
    prior:
      dist: norm
      loc: 0.0191832
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0191832
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^1
  LSST_M2:
    prior:
      dist: norm
      loc: -0.0431752
      scale: 0.005
    ref:
      dist: norm
      loc: -0.0431752
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^2
  LSST_M3:
    prior:
      dist: norm
      loc: -0.034961
      scale: 0.005
    ref:
      dist: norm
      loc: -0.034961
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^3
  LSST_M4:
    prior:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    ref:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^4
  LSST_M5:
    prior:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    ref:
      dist: norm
      loc: -0.0158096
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{LSST}^5
theory:
  emul_cosmic_shear:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args: 
      device: 'cuda'
      file:  ['projects/lsst_y1/emulators/lcdm_nla_halofit_cosmic_shear_trf/transformer.emul']
      extra: ['projects/lsst_y1/emulators/lcdm_nla_halofit_cosmic_shear_trf/transformer.h5']
      ord:   [['logA','ns','H0','omegabh2','omegach2',
               'LSST_DZ_S1','LSST_DZ_S2','LSST_DZ_S3','LSST_DZ_S4','LSST_DZ_S5',
               'LSST_A1_1','LSST_A1_2']]
      extrapar: [{'MLA': 'TRF', 'INT_DIM_RES': 256, 
                  'INT_DIM_TRF': 1024, 'NC_TRF': 32, 'OUTPUT_DIM': 780}]
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
model = get_model(yaml_load(yaml_string))
def likelihood(p):
    point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))
    names=list(model.parameterization.sampled_params().keys())
    point.update({ name: p[name].item() for name in names })
    res1 = model.logprior(point,make_finite=True)
    res2 = model.loglike(point,make_finite=True,cached=False,return_derived=False)
    return res1+res2
def chi2v2(p):
    point = dict(zip(model.parameterization.sampled_params(), p))
    logposterior = model.logposterior(point, as_dict=True)
    chi2likes=-2*np.array(list(logposterior["loglikes"].values()))
    chi2prior=-2*np.atleast_1d(model.logprior(point,make_finite=False))
    return np.concatenate((chi2likes, chi2prior))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def min_chi2(x0,
             cov, 
             fixed=-1, 
             maxfeval=3000, 
             nwalkers=5,
             pool=None):

    def mychi2(params, *args):
        z, fixed, T = args
        params = np.array(params, dtype='float64')
        if fixed > -1:
            params = np.insert(params, fixed, z)
        return chi2(p=params)/T
    if fixed > -1:
        z      = x0[fixed]
        x0     = np.delete(x0, (fixed))
        args = (z, fixed, 1.0)
        cov = np.delete(cov, (fixed), axis=0)
        cov = np.delete(cov, (fixed), axis=1)
    else:
        args = (0.0, -2.0, 1.0)

    def log_prior(params):
        return 1.0
    
    def logprob(params, *args):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return -0.5*mychi2(params, *args) + lp
    
    class GaussianStep:
       def __init__(self, stepsize=0.2):
           self.cov = stepsize*cov
       def __call__(self, x):
           return np.random.multivariate_normal(x, self.cov, size=1)
    
    ndim        = int(x0.shape[0])
    nwalkers    = int(nwalkers)
    nsteps      = maxfeval
    if fixed == -1:
      temperature = np.array([1.0, 0.25, 0.1, 0.005, 0.001], dtype='float64')
    else:
      temperature = np.array([0.25, 0.1, 0.005, 0.001], dtype='float64')
    stepsz      = temperature/4.0

    partial_samples = [x0]
    partial = [mychi2(x0, *args)]

    for i in range(len(temperature)):
        x = [] # Initial point
        for j in range(nwalkers):
            x.append(GaussianStep(stepsize=stepsz[i])(x0)[0,:])
        x = np.array(x,dtype='float64')

        GScov  = copy.deepcopy(cov)
        GScov *= temperature[i]*stepsz[i] 
  
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                        ndim=ndim, 
                                        log_prob_fn=logprob, 
                                        args=(args[0], args[1], temperature[i]),
                                        moves=[(emcee.moves.GaussianMove(cov=GScov),1.)],
                                        pool=pool)
        
        sampler.run_mcmc(x, nsteps, skip_initial_state_check=True)
        samples = sampler.get_chain(flat=True, thin=1, discard=0)

        j = np.argmin(-1.0*np.array(sampler.get_log_prob(flat=True)))
        partial_samples.append(samples[j])
        tchi2 = mychi2(samples[j], *args)
        partial.append(tchi2)
        x0 = copy.deepcopy(samples[j])
        sampler.reset()
    # min chi2 from the entire emcee runs
    j = np.argmin(np.array(partial))
    result = [partial_samples[j], partial[j]]
    return result
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def prf(x0, maxfeval, cov, fixed=-1, nwalkers=5, pool=None):
    t0 = np.array(x0, dtype='float64')
    res =  min_chi2(x0=t0, 
                    fixed=fixed,
                    cov=cov, 
                    maxfeval=maxfeval, 
                    nwalkers=nwalkers,
                    pool=pool)
    return res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        nwalkers = pool.comm.Get_size()
        maxevals = int(args.maxfeval/(4.0*nwalkers))

        # First: load the cov. matrix (from running EXAMPLE_EMUL_MCMC1.yaml) --
        if args.cov is None:
          cov = model.prior.covmat(ignore_external=False) # cov from prior
          factor = min(1.0, args.factor)
        else:
          cov = np.loadtxt(args.root+args.cov)[0:model.prior.d(),0:model.prior.d()]
          factor = args.factor
        sigma = np.sqrt(np.diag(cov))

        # Second: get minimum --------------------------------------------------
        if args.minfile is not None: # load minimum from running MCMC
          x0 = np.loadtxt(args.minfile)
          chi20 = x0[-1]
          x0 = x0[0:model.prior.d()]
        else: # Compute the minimum (slow)
          (x0, results) = model.get_valid_point(max_tries=1000, 
                                     ignore_fixed_ref=False,
                                     logposterior_as_dict=True)
          res = np.array(list(prf(x0=x0, 
                                  maxfeval=int(5.*maxevals/4.), 
                                  nwalkers=nwalkers,
                                  pool=pool,
                                  cov=cov,
                                  fixed=-1)), dtype="object")
          x0 = np.array(res[0], dtype='float64')[0:model.prior.d()]
          chi20 = res[1]
          print(f"Global Min: params = {x0}, and chi2 = {chi20}")
        
        # Third we need to set the parameter profile range ---------------------
        start = np.zeros(model.prior.d(), dtype='float64')
        stop  = np.zeros(model.prior.d(), dtype='float64')
        start = x0 - factor*sigma
        stop  = x0 + factor*sigma
        
        # We need to respect the YAML priors
        bounds0 = model.prior.bounds(confidence=0.999999)
        for i in range(model.prior.d()):
            if (start[i] < bounds0[i][0]):
              start[i] = bounds0[i][0]
            if (stop[i] > bounds0[i][1]):
              stop[i] = bounds0[i][1]

        half_range = (stop[args.profile] - start[args.profile]) / 2.0
       
        numpts = args.numpts-1 if args.numpts%2 == 1 else args.numpts 
      
        param  = np.linspace(start = x0[args.profile] - half_range,
                             stop  = x0[args.profile] + half_range,
                             num = numpts)
        numpts=numpts+1
        param = np.insert(param, numpts//2, x0[args.profile])
        # Print to the terminal ------------------------------------------------
        names = list(model.parameterization.sampled_params().keys()) # Cobaya Call
        print(f"maxfeval={args.maxfeval}, param={names[args.profile]}")
        print(f"profile param values = {param}")
        
        # 4th: we need to set the vectors that will hold the final result ------
        xf = np.tile(x0, (numpts, 1))
        xf[:,args.profile] = param
        
        chi2res = np.zeros(numpts)
        chi2res[numpts//2] = chi20
        # 5th: run from midpoint to right --------------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2+1,numpts):
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      maxfeval=int(maxevals), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res[0], args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = res[1]
            print(f"Partial ({i+1}/{numpts}): params = {tmp}, and chi2 = {chi2res[i]}")
        # 6th: run from midpoint to left ---------------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2-1, -1, -1):
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      maxfeval=int(maxevals), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res[0], args.profile, param[i])
            tmp = np.array(xf[i,:], dtype='float64')
            chi2res[i] = res[1] 
            print(f"Partial ({i+1}/{numpts}): params = {tmp}, and chi2 = {chi2res[i]}")
        # Append derived (begins) ----------------------------------------------
        xf = np.column_stack((xf, 
                              np.array([chi2v2(d) for d in xf], dtype='float64')))
        # --- saving file begins -----------------------------------------------    
        comment = [names[args.profile],"chi2"]+list(model.info()['likelihood'].keys())+["prior"]
        np.savetxt(f"{args.root}chains/{args.outroot}.{names[args.profile]}.txt",
                   np.concatenate([np.c_[param,chi2res],xf],axis=1),
                   fmt="%.6e",
                   header=f"maxfeval={args.maxfeval}, param={names[args.profile]}\n"+' '.join(comment),
                   comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------