## Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment out the following lines in `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>
>     # ------------------------------------------------------------------------------
>     # The keys below control which cosmolike projects will be installed and compiled
>     # ------------------------------------------------------------------------------
>     export IGNORE_COSMOLIKE_LSSTY1_CODE=1
>     #export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
>     (...)
>     # ------------------------------------------------------------------------------
>     # Cosmolike projects below -------------------------------------------
>     # ------------------------------------------------------------------------------
>     (...)
>     export LSST_Y1_URL="https://github.com/CosmoLike/cocoa_lsst_y1.git"
>     export LSST_Y1_NAME="lsst_y1"
>     #BRANCH: if unset, load the latest commit on the specified branch
>     #export LSST_Y1_BRANCH="dev"
>     #COMMIT: if unset, load the specified commit
>     export LSST_Y1_COMMIT="1abe548281296196dabee7b19e31c56f324eda38"
>     #TAG: if unset, load the specified TAG
>     #export LSST_Y1_TAG="v4.0-beta17"

> [!NOTE]
> In case users need to rerun `setup_cocoa.sh`, Cocoa will not download previously installed packages, cosmolike projects, or large datasets, unless the following keys are set on `set_installation_options.sh`
>
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # ------------------------------------------------------------------------------
>     # OVERWRITE_EXISTING_XXX_CODE=1 -> setup_cocoa overwrites existing PACKAGES ----
>     # overwrite: delete the existing PACKAGE folder and install it again -----------
>     # redownload: delete the compressed file and download data again ---------------
>     # These keys are only relevant if you run setup_cocoa multiple times -----------
>     # ------------------------------------------------------------------------------
>     (...)
>     export OVERWRITE_EXISTING_ALL_PACKAGES=1    # except cosmolike projects
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 # dangerous (possible loss of uncommitted work)
>                                                 # if unset, users must manually delete cosmolike projects
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1      # warning: some data is many GB
>

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh
> 
> or (in case users just want to compile the lsst-y1 project)
>
>       source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
> 
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project lsst-y1 contains jupyter notebook examples located at `projects/lsst_y1`.

To run the example

 **Step :one:**: activate the Cocoa Conda environment,  and the private Python environment 
    
      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).
    
    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8; export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE
      
 **Step :three:**: The folder `projects/lsst_y1` contains examples. So, run the `cobaya-run` on the first example following the commands below.


- **One model evaluation**:

      mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
         --bind-to core:overload-allowed --mca mpi_yield_when_idle 1 --report-bindings  \
         --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
         cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

      mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
         --bind-to core:overload-allowed --mca mpi_yield_when_idle 1 --report-bindings  \
         --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
         cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f

# Running ML emulators <a name="cobaya_base_code_examples_emul"></a>

Cocoa contains a few transformer- and CNN-based neural network emulators capable of simulating CMB, cosmolike, matter power spectrum, and distances. We provide a few scripts that exemplify their API. To run them, users must have commented out the following lines on `set_installation_options.sh` before running the `setup_cocoa.sh` and `compile_cocoa.sh`.

      [Adapted from Cocoa/set_installation_options.sh shell script] 
      # insert the # symbol (i.e., unset these environmental keys  on `set_installation_options.sh`)
      #export IGNORE_EMULTRF_CODE=1              #SaraivanovZhongZhu (SZZ) transformer/CNN-based emulators
      #export IGNORE_EMULTRF_DATA=1            
      #export IGNORE_LIPOP_LIKELIHOOD_CODE=1     # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_LIPOP_CMB_DATA=1           
      #export IGNORE_ACTDR6_CODE=1               # to run EXAMPLE_EMUL_(EVALUATE/MCMC/NAUTILUS/EMCEE1).yaml
      #export IGNORE_ACTDR6_DATA=1         
      #export IGNORE_NAUTILUS_SAMPLER_CODE=1     # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_NAUTILUS1.py
      #export IGNORE_POLYCHORD_SAMPLER_CODE=1    # to run PROJECTS/EXAMPLE/EXAMPLE_EMUL_POLY1.yaml
      #export IGNORE_GETDIST_CODE=1              # to run EXAMPLE_TENSION_METRICS.ipynb
      #export IGNORE_TENSIOMETER_CODE=1          # to run EXAMPLE_TENSION_METRICS.ipynb
      
> [!TIP]
> What if users have not configured ML-related keys before sourcing `setup_cocoa.sh`?
> 
> Answer: comment the keys below before rerunning `setup_cocoa.sh`.
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     # These keys are only relevant if you run setup_cocoa multiple times
>     #export OVERWRITE_EXISTING_ALL_PACKAGES=1    
>     #export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 
>     #export REDOWNLOAD_EXISTING_ALL_DATA=1

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Ensure OpenMP is **OFF**.
    
    export OMP_NUM_THREADS=1
    
 **Step :three:** Run `cobaya-run` on the first emulator example following the commands below.

- **One model evaluation**:

      mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

      mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC1.yaml -f

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1`)

      mpirun -n 4 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC2.yaml -f
  
> [!Note]
> The examples below may require a large number of MPI workers. Before running them, it may be necessary to increase 
> the limit of threads that can be created (at UofA HPC type `ulimit -u 1000000`), otherwise users 
> may encounter the error `libgomp: Thread creation failed`

- **PolyChord**:

      mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY1.yaml -f

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - 38 parameters)

      mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY2.yaml -f
          
> [!Note]
> The `Nautilis`, `Minimizer`, `Profile`, and `Emcee` scripts below contain an internally defined `yaml_string` that specifies priors, 
> likelihoods, and the theory code, all following Cobaya Conventions.

- **Nautilus**:

      mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
              --root ./projects/lsst_y1/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
              --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - 38 parameters)

      mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS2.py \
              --root ./projects/lsst_y1/ --outroot "EXAMPLE_EMUL_NAUTILUS2"  \
              --maxfeval 850000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5
        
- **Emcee**:

      mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE1.py --root ./projects/lsst_y1/ \
              --outroot "EXAMPLE_EMUL_EMCEE1" --maxfeval 500000 --burn_in 0.3

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - 38 parameters)

      mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
        --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
        python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE2.py --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE2" --maxfeval 1400000 --burn_in 0.3
      
  The number of steps per MPI worker is $n_{\\rm sw} =  {\\rm maxfeval}/n_{\\rm w}$,
  with the number of walkers being $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.
  For proper convergence, each walker should traverse 50 times the auto-correlation length,
  which is provided in the header of the output chain file.
  
  The scripts that made the plots below are provided at `projects/lsst_y1/script/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[2].py`

<p align="center">
<img width="750" height="750" alt="Screenshot 2025-08-03 at 4 19 17 PM" src="https://github.com/user-attachments/assets/fe4c4dd8-ec60-43d9-bc15-a297f67bd620" />
</p>

<p align="center">
<img width="750" height="750" alt="Screenshot 2025-08-03 at 4 19 17 PM" src="https://github.com/user-attachments/assets/fe4c4dd8-ec60-43d9-bc15-a297f67bd620" />
</p>

- **Global Minimizer**:

      mpirun -n 51 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE1.py --root ./projects/lsst_y1/ \
              --outroot "EXAMPLE_EMUL_MIN1" --nstw 250
  
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - 38 parameters)

      mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
          --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE2.py --root ./projects/lsst_y1/ \
              --outroot "EXAMPLE_EMUL_MIN2" --nstw 250

  The number of steps per Emcee walker per temperature is $n_{\\rm stw}$,
  and the number of walkers is $n_{\\rm w}={\\rm max}(3n_{\\rm params},n_{\\rm MPI})$.

  The script of the plot below is provided at `projects/lsst_y1/script/EXAMPLE_PLOT_MIN_COMPARE_CONV[2].py`

<p align="center">
<img width="750" height="750" alt="Screenshot 2025-08-12 at 8 36 33 PM" src="https://github.com/user-attachments/assets/31c36592-2d6c-4232-b5b4-5f686f9f2b8e" />
</p>

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - 38 parameters)

<p align="center">
<img width="750" height="750" alt="Screenshot 2025-08-13 at 5 29 59 PM" src="https://github.com/user-attachments/assets/12130055-9697-4326-8ffe-83654e9b564d" />
</p>
