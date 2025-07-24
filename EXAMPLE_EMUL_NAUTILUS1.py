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
    message=r".*overflow encountered in exp.*"
)
import argparse, random
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from nautilus import Prior, Sampler
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_NAUTILUS1')
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
                    default="example_nautilus1")
parser.add_argument("--nlive",
                    dest="nlive",
                    help="Number of live points ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1000)
parser.add_argument("--maxfeval",
                    dest="maxfeval",
                    help="Minimizer: maximum number of likelihood evaluations",
                    type=int,
                    nargs='?',
                    const=1,
                    default=100000)
parser.add_argument("--neff",
                    dest="neff",
                    help="Minimum effective sample size. ",
                    type=int,
                    nargs='?',
                    const=1,
                    default=10000)
parser.add_argument("--flive",
                    dest="flive",
                    help="Maximum fraction of the evidence contained in the live set before building the initial shells terminates",
                    type=float,
                    nargs='?',
                    const=1,
                    default=0.01)
args, unknown = parser.parse_known_args()
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
  ns:
    prior:
      min: 0.92
      max: 1.05
  H0:
    prior:
      min: 55
      max: 91
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
  omegam:
    prior:
      min: 0.1
      max: 0.9
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
info  = yaml_load(info_txt)
model = get_model(info)
def likelihood(p):
    point = dict(zip(model.parameterization.sampled_params(),
                 model.prior.sample(ignore_external=True)[0]))
    names=list(model.parameterization.sampled_params().keys())
    point.update({ name: p[name].item() for name in names })
    logposterior = model.logposterior(point, as_dict=True,make_finite=True)
    res1 = model.logprior(point,make_finite=False)
    res2 = model.loglike(point,make_finite=False,cached=False,return_derived=False)
    return res1+res2
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
from mpi4py.futures import MPIPoolExecutor
if __name__ == '__main__':
    #rnd = random.randint(0,1000)
    rnd = 642
    cfile= args.root + "chains/" + args.outroot +  "_" + str(rnd) + "_checkpoint" ".hdf5"
    print(f"nlive={args.nlive}, output={cfile}")
    # Here we need to pass Cobaya Prior to Nautilus
    NautilusPrior = Prior()                              # Nautilus Call 
    dim    = model.prior.d()                             # Cobaya call
    bounds = model.prior.bounds(confidence=0.999999)     # Cobaya call
    names  = list(model.parameterization.sampled_params().keys()) # Cobaya Call
    for b, name in zip(bounds, names):
      NautilusPrior.add_parameter(name, dist=(b[0], b[1]))
    # Start Nautilus
    sampler = Sampler(NautilusPrior, 
                      likelihood,  
                      filepath=cfile, 
                      n_dim=dim,
                      pool=MPIPoolExecutor(),
                      n_live=args.nlive,
                      resume=True)
    sampler.run(f_live=args.flive,
                n_eff=args.neff,
                n_like_max=args.maxfeval,
                verbose=True,
                discard_exploration=True)
    points, log_w, log_l = sampler.posterior()
    # --- saving file begins --------------------
    cfile   = args.root + "chains/" + args.outroot +  "_" + str(rnd) + ".txt"
    print("Output file = ", cfile)
    np.savetxt(cfile,
               np.column_stack((np.exp(log_w), log_l, points, -2*log_l)),
               header=f"nlive={args.nlive}, output={cfile}",
               comments="# ")
    # --- saving file ends --------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#HOW TO CALL THIS SCRIPT
#mpirun -n 5 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
#   --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
#   python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
#   --root ./projects/lsst_y1/ --outroot "example_nautilus1"  \
#   --maxfeval 50000 --nlive 500 --neff 10000 --flive 0.01