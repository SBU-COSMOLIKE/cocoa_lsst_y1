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
maxfeval = args.maxfeval
oroot    = args.root + "chains/" + args.outroot
index    = args.profile
numpts   = args.numpts
nwalkers = args.nwalkers
cov_file = args.root + args.cov
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
info_txt = r"""
likelihood:
  lsst_y1.cosmic_shear:
    path: ./external_modules/data/lsst_y1
    data_file: lsst_y1_M1_GGL0.05.dataset
    use_emulator: True
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
model = get_model(yaml_load(info_txt))
name = list(model.parameterization.sampled_params().keys())
def chi2(p):
    point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))
    names = list(model.parameterization.sampled_params().keys())
    point.update({name: val for name, val in zip(names, p)})
    res1 = model.logprior(point,make_finite=False)
    res2 = model.loglike(point,make_finite=False,cached=False,return_derived=False)
    return -2.0*(res1+res2)
def chi2v2(p):
    point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))
    names=list(model.parameterization.sampled_params().keys())
    point.update({name: val for name, val in zip(names, p)})
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
             maxiter=10,
             nwalkers=5):
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
    temperature = np.array([1.0, 0.25, 0.1, 0.005, 0.001], dtype='float64')
    stepsz      = temperature/4.0

    mychi2(x0, *args) # first call takes a lot longer (when running on cuda)
    start_time = time.time()
    mychi2(GaussianStep(stepsize=0.1)(x0)[0,:], *args)
    elapsed_time = time.time() - start_time
    print(f"Emcee: nwalkers = {nwalkers}, "
          f"nTemp = {len(temperature)}, "
          f"feval (per walker) = {maxfeval}, "
          f"feval (per Temp) = {nwalkers*maxfeval}, "
          f"feval = {nwalkers*maxfeval*len(temperature)}")
    print(f"Emcee: Like Eval Time: {elapsed_time:.4f} secs, "
          f"Eval Time = {elapsed_time*nwalkers*maxfeval*len(temperature)/3600.:.4f} hours.")

    partial_samples = []
    partial = []

    for i in range(len(temperature)):
        x = [] # Initial point
        for j in range(nwalkers):
            x.append(GaussianStep(stepsize=stepsz[i])(x0)[0,:])
        x = np.array(x,dtype='float64')

        GScov  = copy.deepcopy(cov)
        GScov *= temperature[i]*stepsz[i] 
  
        sampler = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        logprob, 
                                        args=(args[0], args[1], temperature[i]),
                                        moves=[(emcee.moves.GaussianMove(cov=GScov),1.)])
        
        sampler.run_mcmc(x, nsteps, skip_initial_state_check=True)
        samples = sampler.get_chain(flat=True, thin=1, discard=0)

        j = np.argmin(-1.0*np.array(sampler.get_log_prob(flat=True)))
        partial_samples.append(samples[j])
        tchi2 = mychi2(samples[j], *args)
        partial.append(tchi2)
        x0 = copy.deepcopy(samples[j])
        sampler.reset()
        #print(f"emcee: i = {i}, chi2 = {tchi2}, param = {args[0]}")
    # min chi2 from the entire emcee runs
    j = np.argmin(np.array(partial))
    return [partial_samples[j], partial[j]]

def prf(x0, index, maxfeval, cov, nwalkers=5, maxiter=1):
    t0 = np.array(x0, dtype='float64')
    res =  min_chi2(x0=t0, 
                    fixed=index, 
                    maxfeval=maxfeval, 
                    maxiter=maxiter,
                    nwalkers=nwalkers,
                    cov=cov)
    return res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from mpi4py.futures import MPIPoolExecutor

if __name__ == '__main__':
    # First we need to load minimum (from running EXAMPLE_EMUL_MINIMIZE1.py)
    x = np.loadtxt(args.minfile)[0:model.prior.d()]
    
    cov = np.loadtxt(cov_file)[0:model.prior.d(),0:model.prior.d()]
    sigma = np.sqrt(np.diag(cov))
    start = np.zeros(len(x), dtype='float64')
    stop  = np.zeros(len(x), dtype='float64')
    start = x - args.factor*sigma
    stop  = x + args.factor*sigma
    bounds0 = model.prior.bounds(confidence=0.999999)
    for i in range(len(x)):
        if (start[i] < bounds0[i][0]):
          start[i] = bounds0[i][0]
        if (stop[i] > bounds0[i][1]):
          stop[i] = bounds0[i][1]
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    print(f"nwalkers={nwalkers}, maxfeval={maxfeval}, param={index}")
    param = np.linspace(start=start[index], stop=stop[index], num=numpts)
    print(f"profile param values = {param}")

    x0 = np.tile(x,(param.size,1))
    x0[:,index] = param
    pool = MPIPoolExecutor()
    res = np.array(list(pool.map(functools.partial(prf, 
                                                   index=index, 
                                                   maxfeval=maxfeval, 
                                                   nwalkers=nwalkers,
                                                   cov=cov),x0)),dtype="object")
    x0 = np.array([np.insert(r,index,p) for r, p in zip(res[:,0],param)],dtype='float64')
    
    # Append individual chi2 (in case there are more than one data) (begins) ---
    tmp = np.array([chi2v2(d) for d in x0], dtype='float64')
    x0 = np.column_stack((x0,tmp[:,0]))
    # Append individual chi2 (in case there are more than one data) (ends) -----
    # --- saving file begins ---------------------------------------------------
    names = list(model.parameterization.sampled_params().keys()) # Cobaya Call
    names = [names[index], "chi2"]+names+list(model.info()['likelihood'].keys())
    rnd = random.randint(0,1000)
    out = oroot + "_" + str(rnd) + "_" + name[index] 
    print("Output file = ", out + ".txt")
    np.savetxt(out+".txt",
               np.concatenate([np.c_[param,res[:,1]],x0],axis=1),
               fmt="%.6e",
               header=f"nwalkers={nwalkers}, maxfeval={maxfeval}, param={name[index]}\n"+' '.join(names),
               comments="# ")
    # --- saving file ends -----------------------------------------------------
    pool.shutdown()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#HOW TO CALL THIS SCRIPT
#export nmpi=5
#export numpts=$((${nmpi}-1))
#mpirun -n ${nmpi} --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#  --bind-to core:overload-allowed --mca mpi_yield_when_idle 1 \
#  --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
#  python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE1.py \
#  --cov 'EXAMPLE_EMUL_MCMC1.covmat' --nwalkers 5 --numpts ${numpts} \
#  --profile ${SLURM_ARRAY_TASK_ID} --maxfeval 9000 --factor 3 \
#  --minfile="./projects/lsst_y1/example_min1_lcdm.txt"