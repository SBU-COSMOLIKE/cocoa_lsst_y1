# LSST-Y1 COLA Chains
This branch of the `lsst_y1` repository contains yaml files for LSST-Y1 chains to be run in Cocoa, using the COLA emulators as well as Euclid Emulator 2.

## Description
All chains run with LSST-Y1 Cosmic Shear, EE2 fiducial data vector, same nuisance parameter priors (pessimistic IA NLA, optimistic photo-z and shear calibration). The priors for cosmological parameters are the EE2 box boundaries. We are only considering wCDM. The convergence criterion is given by `R-1 < 0.005`.

The variables are: masks (scale cuts), the emulators (EE2, COLA, COLA+10, COLA+25, COLA+100, where COLA+X means COLA with X anchors).

THe fiducial data vector `data/EE2_FIDUCIAL.modelvector` was generated with the yaml file `CREATE_EE2_FIDUCIAL_DATA_VECTOR.yaml`. A noise realization was added to the data vector, using the Python script `add_noise_to_data_vector.py`, resulting in the file `data/EE2_FIDUCIAL_NOISED.modelvector`. For each mask, one `dataset` file was created, in `data/LSST_Y1_MX_EE2_FID.dataset`. For testing purposes, an additional dataset `LSST_Y1_M6_EE2_NO_NOISE.dataset` was created using the noiseless data vector. The noiseless dataset is used in `TEST_COLA_EMU_SHEAR.yaml`

The nonlinear emulators are set in the yaml file under the `non_linear_emul` option. They are numbered as:

1 - EE2;

2 - Halofit;

3 - GP;

4 - NN;

5 - PCE;

More emulators will be added since we need to implement the different anchors, COLA precision, dark energy models...

**NOTE**: There are still some placeholders in the yaml files, namely `COLA_EMU_NO_ANCHOR`, `COLA_EMU_10_ANCHORS`, `COLA_EMU_25_ANCHORS` and `COLA_EMU_100_ANCHORS`. Before running chains, these strings need to be substituted by the actual numbers representing the chosen emulator.

## List
1 - EE2, M1

2 - EE2, M2

3 - EE2, M3

4 - EE2, M4

5 - EE2, M5

6 - COLA, M1

7 - COLA, M2

8 - COLA, M3

9 - COLA, M4

10 - COLA, M5

11 - COLA+10, M1

12 - COLA+10, M2

13 - COLA+10, M3

14 - COLA+10, M4

15 - COLA+10, M5

16 - COLA+25, M1

17 - COLA+25, M2

18 - COLA+25, M3

19 - COLA+25, M4

20 - COLA+25, M5

21 - COLA+100, M1

22 - COLA+100, M2

23 - COLA+100, M3

24 - COLA+100, M4

25 - COLA+100, M5


## Running Cosmolike projects <a name="running_cosmolike_projects"></a> 

In this tutorial, we assume the user installed Cocoa via the *Conda installation* method, and the name of the Conda environment is `cocoa`. We also presume the user's terminal is in the folder where Cocoa was cloned.

:one: **Step 1 of 6**: activate the Conda Cocoa environment
    
        $ conda activate cocoa

:two: **Step 2 of 6**: go to the `projects` folder and clone the Cosmolike LSST-Y1 project:
    
        $(cocoa) cd ./cocoa/Cocoa/projects
        $(cocoa) git clone --depth 1 git@github.com:CosmoLike/cocoa_lsst_y1.git lsst_y1

The option `--depth 1` prevents git from downloading the entire project history. By convention, the Cosmolike Organization hosts a Cobaya-Cosmolike project named XXX at `CosmoLike/cocoa_XXX`. However, our scripts and YAML files assume the removal of the `cocoa_` prefix when cloning the repository.
 
:three: **Step 3 of 6**: go back to the Cocoa main folder, and activate the private Python environment
    
        $(cocoa) cd ../
        $(cocoa) source start_cocoa
 
:warning: (**warning**) :warning: Remember to run the start_cocoa script only after cloning the project repository. The script *start_cocoa* creates the necessary symbolic links and adds the *Cobaya-Cosmolike interface* of all projects to `LD_LIBRARY_PATH` and `PYTHONPATH` paths.

:four: **Step 4 of 6**: compile the project
 
        $(cocoa)(.local) source ./projects/lsst_y1/scripts/compile_lsst_y1

:five: **Step 5 of 6**: select the number of OpenMP cores
    
        $(cocoa)(.local) export OMP_PROC_BIND=close; export OMP_NUM_THREADS=4
        
:six:  **Step 6 of 6**: run a template YAML file

One model evaluation:

        $(cocoa)(.local) mpirun -n 1 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f
 
MCMC:

        $(cocoa)(.local) mpirun -n 4 --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f

## Deleting Cosmolike projects <a name="running_cosmolike_projects"></a>

:warning: (**warning**) :warning: Never delete the `lsst_y1` folder from the project folder without running `stop_cocoa` first; otherwise, Cocoa will have ill-defined soft links at `Cocoa/cobaya/cobaya/likelihoods/`, `Cocoa/external_modules/code/` and `Cocoa/external_modules/data/`

## MCMC Convergence Criteria <a name="running_cosmolike_projects"></a>

  We are strict in our convergence criteria on `EXAMPLE_MCMC[0-9].YAML` MCMC examples.
  
    Rminus1_stop: 0.005
    # Gelman-Rubin R-1 on std deviations
    Rminus1_cl_stop: 0.15
    
For most applications, these settings are overkill (except when computing some tension and goodness of fit metrics). Please adjust these settings to your needs. 
