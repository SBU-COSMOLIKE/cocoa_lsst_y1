# COLA $w_0w_a$
*In case of bugs, contact João ASAP!*

This repository contains a simulated LSST analysis using COLA emulators as nonlinear prescription, comparing the results with those obtained using EE2.

**NOTE**: Users must have installed the Cocoa environment with the machine learning libraries, so you must unset the `IGNORE_EMULATOR_CPU_PACKAGES` flag inside the `set_installation_options.sh` file before running the `compile_cocoa.sh` command. If that's not the case, you can run the compile command again.

**NOTE**: I had to make a small hack in the `keras.saving` library in order to load the NN models properly. Contact João for more information!

## Usage

See the `TEST_COLA.yaml` file that calculates one evaluation of the LSST cosmic shear likelihood for the reference cosmology using the COLA emulator, assuming the EE2 reference cosmology for the fiducial data vector (see below).

## Fiducial Data Vectors and Dataset files

The fiducial data vectors are stored in `data/`. Their names have the format `<nonlinear_prescription>_<fiducial_cosmology>.modelvector`.

We define a few fiducial cosmologies:
- EE2REF: EuclidEmulator2 reference cosmology: $\Omega_m = 0.319$, $\Omega_b = 0.049$, $h = 0.67$, $A_s = 2.1\times 10^{-9}$, $n_s = 0.96$, $w = -1$, $w_a = 0$
- DESI2CMBPANPLUS: $\Omega_m = 0.3114$, $\Omega_b = 0.049$, $h = 0.6751$, $A_s = 2.1\times 10^{-9}$, $n_s = 0.96$, $w = -0.838$, $w_a = -0.62$ (values from Table V of [DESI DR2](https://arxiv.org/pdf/2503.14738))

The nonlinear prescriptions are `ee2` and `cola`.

The `.dataset` files contain information about the fiducial data vector as well as the scale cuts used. Therefore, these files have the format `<nonlinear>_<fid_cosmo>_<scale_cuts>.dataset`. Other fields in this file are going to be kept fixed.

## Checking $\Delta \chi^2$

Inside `data/`, there is a Python notebook `check_delta_chi2.ipynb` that calculates the $\Delta \chi^2$ between two data vectors.

## Chains

See `CHAINS.md` for the list of chains. The Python script `generate_yamls.py` takes the `CHAINS.md` file as input and generates all of the yaml files. This ensures that the yamls are consistent, decreases the chance of errors and promotes a documentation-first approach, ensuring that all chains are properly listed.

## Emulator implementation in Cocoa

See `likelihood/_cosmolike_prototype_base.py` for the implementation.

# Original Docs
## Running Cosmolike projects <a name="running_cosmolike_projects"></a> 

In this tutorial, we assume the user installed Cocoa via the *Conda installation* method, and the name of the Conda environment is `cocoa`. We also presume the user's terminal is in the folder where Cocoa was cloned.

 **Step :one:**: activate the cocoa Conda environment, go to the `cocoa/Cocoa/projects` folder, and clone the Cosmolike LSST-Y1 project:
    
      conda activate cocoa

and

      cd ./cocoa/Cocoa/projects

and

      ${CONDA_PREFIX}/bin/git clone --depth 1 https://github.com/CosmoLike/cocoa_lsst_y1.git --branch v4.0-beta5 lsst_y1 

:warning: Cocoa scripts and YAML files assume the removal of the `cocoa_` prefix when cloning the repository.

:interrobang: If the user is a developer, then type the following instead *(at your own risk!)*

      ${CONDA_PREFIX}/bin/git clone git@github.com:CosmoLike/cocoa_lsst_y1.git lsst_y1
      
 **Step :two:**: go back to the Cocoa main folder and activate the private Python environment
    
      cd ../

and

      source start_cocoa.sh
 
:warning: Remember to run the `start_cocoa.sh` shell script only **after cloning** the project repository (or if you already in the `(.local)` environment, run `start_cocoa.sh` again). 

**Step :three:**: compile the project
 
      source ./projects/lsst_y1/scripts/compile_lsst_y1

:interrobang: The script `compile_cocoa.sh` also compiles every Cosmolike project on the `cocoa/Cocoa/projects/` folder.

**Step :four:**: select the number of OpenMP cores (below, we set it to 4), and run a template YAML file

    
      export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8
      
One model evaluation:
      
      mpirun -n 1 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f
 
MCMC:

      mpirun -n 4 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f

Profile:

      cd ./projects/lsst_y1

and

      export NMPI=4

and

      mpirun -n ${NMPI} --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} python -m mpi4py.futures EXAMPLE_PROFILE1.py --mpi $((${NMPI}-1)) --profile 1 --tol 0.05 --AB 1.0 --outroot 'profile' --minmethod 5 --maxiter 2 --maxfeval 500 

## Deleting Cosmolike projects <a name="running_cosmolike_projects"></a>

Do not delete the `lsst_y1` folder from the project folder without first running the shell script `stop_cocoa.sh`. Otherwise, Cocoa will have ill-defined soft links. 

:interrobang: Where the ill-defined soft links will be located? 
     
     Cocoa/cobaya/cobaya/likelihoods/
     Cocoa/external_modules/code/
     Cocoa/external_modules/data/ 
    
:interrobang: Why does Cocoa behave like this? The shell script `start_cocoa.sh` creates symbolic links so Cobaya can see the likelihood and data files. Cocoa also adds the Cobaya-Cosmolike interface of all cosmolike-related projects to the `LD_LIBRARY_PATH` and `PYTHONPATH` environmental variables.

## MCMC Convergence Criteria <a name="running_cosmolike_projects"></a>

  We are strict in our convergence criteria on `EXAMPLE_MCMC[0-9].YAML` MCMC examples.
  
    Rminus1_stop: 0.005
    # Gelman-Rubin R-1 on std deviations
    Rminus1_cl_stop: 0.15
    
These settings are overkill for most applications, except when computing some tension and goodness of fit metrics. Please adjust these settings to your needs. 
