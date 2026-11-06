# Table of contents <a name="table_of_contents"></a>

1. [Reading path through the examples](#reading_path)
2. [Running Cosmolike projects (Basic instructions)](#running_cosmolike_projects)
3. [Baryonic feedback on EXAMPLE_EVALUATE1 and EXAMPLE_EVALUATE3](#lsst_y1_baryonic_feedback)
4. [Running ML emulators](#cobaya_base_code_examples_emul)
5. [Running Hybrid Cosmolike-ML emulators](#cobaya_base_code_examples_emul2)
6. [Running Fisher](#lsst_examples_fisher)
7. [Unit tests](#unit_tests)
8. [Computing covariances](#computing_covariances)
9. [Exploring notebooks](#notebooks)
10. [Appendix: Which accuracy settings are available?](#accuracy)
11. [Appendix: Which accuracy settings should I choose?](#accuracy_choice)

# Reading path through the examples <a name="reading_path"></a>

Start with one evaluation of `EXAMPLE_EVALUATE1.yaml`
([basic instructions](#running_cosmolike_projects)): it checks the
installation that every later example uses.

```mermaid
flowchart TB
  A["EXAMPLE_EVALUATE1.yaml: first evaluation"] --> B["EXAMPLE_EVALUATE1.ipynb: cosmic shear and Fisher"]
  B --> C["EXAMPLE_EVALUATE2.ipynb: lensing and clustering"]
  C --> D["EXAMPLE_EVALUATE3.ipynb: bfmt feedback"]
  C --> E["EXAMPLE_EVALUATE_COVARIANCE.ipynb: covariance terms"]
  E --> F["compute_covariance.py: production covariance"]
  A --> G["EXAMPLE_MCMC1.yaml and EXAMPLE_EMUL_*: chains and minima"]
  G --> H["EXAMPLE_EMUL_PROFILE*.py: profiles"]
  G --> I["scripts/EXAMPLE_PLOT_*.py: figures"]
  H --> I
  A --> J["EXAMPLE_EMUL2_*: hybrid examples"]
```

- The three data-vector notebooks share one fiducial point.
  `EXAMPLE_EVALUATE1.ipynb` introduces the wrapper calls on cosmic shear,
  `EXAMPLE_EVALUATE2.ipynb` adds the lens probes of the 3×2pt data
  vector, and `EXAMPLE_EVALUATE3.ipynb` applies the `bfmt` feedback
  models to that vector. The last one needs `bfmt` and its emulators
  installed ([baryonic feedback](#lsst_y1_baryonic_feedback)).
- The covariance notebook and the production CLI need the optional
  covariance build ([computing covariances](#computing_covariances)).
- The CosmoLike chain and the emulator samplers write chains and minima
  under `chains/`. The profile scripts read a minimum (`--minfile`) and,
  optionally, a chain covariance (`--cov`); the plotting scripts read the
  chains, minima and profiles ([ML emulators](#cobaya_base_code_examples_emul)).
- The hybrid examples ([hybrid emulators](#cobaya_base_code_examples_emul2))
  use neither the data-vector emulators nor those chains.

# Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> `setup_cocoa.sh` and `compile_cocoa.sh` install and compile the cosmolike projects selected in `set_installation_options.sh`. A commented `IGNORE_COSMOLIKE_*_CODE` key enables a project; an active key skips it. The defaults enable LSST Y1, DES × Planck and Roman real.
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     #export IGNORE_COSMOLIKE_LSST_Y1_CODE=1
>     export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     #export IGNORE_COSMOLIKE_DESXPLANCK_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     #export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_KL_CODE=1
>     export IGNORE_COSMOLIKE_DES_CLUSTER_CODE=1
>     # WARNING: des_y6 is not production ready.
>     export IGNORE_COSMOLIKE_DES_Y6_CODE=1
>     (...)
>     export LSST_Y1_URL="https://github.com/CosmoLike/cocoa_lsst_y1.git"
>     export LSST_Y1_NAME="lsst_y1"
>     export LSST_Y1_GIT_TAG="v5.05"
>
> Each released project is pinned to a tag. To select another revision, set only one of its `GIT_COMMIT`, `GIT_BRANCH` or `GIT_TAG` keys (here `LSST_Y1_GIT_COMMIT`, `LSST_Y1_GIT_BRANCH` or `LSST_Y1_GIT_TAG`): a commit takes precedence over a branch, and a branch over a tag. With none set, Cocoa clones the repository's default branch.

> [!NOTE]
> Rerunning `setup_cocoa.sh` (or `./installation_scripts/setup_cosmolike_projects.sh`) keeps an existing `projects/lsst_y1` folder, so uncommitted work survives. To replace the folder with a fresh clone, add the key below to `set_installation_options.sh` before rerunning; it deletes the folder first.
>
>     export OVERWRITE_EXISTING_COSMOLIKE_CODE=1 # dangerous (possible loss of uncommitted work)
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

> [!NOTE]
> The example notebooks load their shared support functions from
> `Cocoa/external_modules/code/cosmolike_core/cosmolike_notebook_utils/`:
> the CAMB run packaged for cosmolike, the data-vector plots, and the
> Fisher-forecast helpers, called through the `cnu` namespace. What is
> specific to this project lives in
> `interface/cosmolike_lsst_y1_notebook_wrappers.py`, imported as `nw`:
> the fiducial point and the probe, $`\chi^2`$, Fisher and `bfmt`
> wrappers around the compiled-interface calls.

To run the example

 **Step :one:**: activate the Cocoa Conda environment,  and the private Python environment 

      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (the commands below set it to 8).

  - Linux
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)
    
        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: The folder `projects/lsst_y1` contains examples. So, run the `cobaya-run` on the first example following the commands below.

> [!Warning] 
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f

  - macOS (arm)
     
        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_MCMC1.yaml -f


# Baryonic feedback on EXAMPLE_EVALUATE1 and EXAMPLE_EVALUATE3 <a name="lsst_y1_baryonic_feedback"></a>

`EXAMPLE_EVALUATE1.yaml` can apply an external baryonic feedback suppression to the
matter power spectrum via the `bfmt` theory block (SP(k), BCEmu, Flamingo, BACCOemu,
or BCemu2025). By default, the example runs without feedback. The notebook
`EXAMPLE_EVALUATE3.ipynb` compares these models on the 3×2pt data vector
([below](#lsst_y1_feedback_notebook)).

**Step :one:**: ensure the lines below are commented out in `set_installation_options.sh`
before running `setup_cocoa.sh` and `compile_cocoa.sh`. *By default, these lines should
be commented out, but it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_PYSPK_CODE=1     # SP(k)
      #export IGNORE_BCEMU_CODE=1     # BCEmu and BCemu2025
      #export IGNORE_FBRE_CODE=1      # FlamingoBaryonResponseEmulator
      #export IGNORE_BACCOEMU_CODE=1  # BACCOemu
      #export IGNORE_BFMT_CODE=1      # Baryon Feedback Theory Block

**Step :two:**: in `EXAMPLE_EVALUATE1.yaml`, uncomment the `bfmt` theory block and select
the model:

      theory:
        bfmt:
          baryon_model: 2 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu, 5 = BCemu2025

**Step :three:**: set `external_baryon_suppression: True` on the `lsst_y1.cosmic_shear`
likelihood block.

**Step :four:**: uncomment the selected model's parameters in the `params` block and in
the `sampler: evaluate: override` block (the example carries a commented block for each
model).

> [!NOTE]
> BACCOemu (`baryon_model: 4`) rejects this example's fiducial `omegab: 0.04`, the edge
> of its baryon-density training box. Move `omegab` inside that box before selecting it;
> the feedback tests evaluate BACCOemu at `omegab: 0.049`.

> [!TIP]
> For the sampled parameters of each model, their validity ranges, and the `bfmt`
> options, see `Cocoa/external_modules/code/baryon_suppression/README.md`.

## The feedback notebook <a name="lsst_y1_feedback_notebook"></a>

[EXAMPLE_EVALUATE3.ipynb](EXAMPLE_EVALUATE3.ipynb) drives the same `bfmt` block for the
three SP(k) relations, BCEmu, Flamingo and BCemu2025 at the example's fiducial point;
BACCOemu is left out for the reason above. The notebook builds its own Cobaya models, so it needs the installation keys of
Step one but none of the YAML edits. For each model it applies the suppression
$`S(k,z)`$ to the nonlinear matter power and plots the ratio to the feedback-free
prediction for $`C_\ell^{EE}`$, $`\xi_\pm`$, $`C_\ell^{gs}`$ and $`\gamma_t`$. A final
section sweeps each model's parameters one at a time and runs CAMB about 130 times
(tens of minutes).

Its section 6 prints the $\chi^2$ of the feedback-free prediction and of each model's
prediction against the stored feedback-free 3×2pt data vector under the M1 mask.
A model's $\chi^2$ well above the feedback-free one means the M1 scale cuts do not
protect an LSST Y1 analysis from feedback of that strength.

# Running ML emulators <a name="cobaya_base_code_examples_emul"></a>

Cocoa contains a few transformer- and CNN-based neural network emulators capable of simulating the CMB, cosmolike outputs, matter power spectrum, and distances. The scripts below exemplify their API. To run them, keep the following lines commented out in `set_installation_options.sh` before running `setup_cocoa.sh` and `compile_cocoa.sh`; they are commented out by default. The examples marked `Planck CMB (l < 396) + SN + BAO + LSST-Y1` combine LSST Y1 cosmic shear with Planck 2018, DES Y5 supernovae and DESI DR2 BAO.

      [Adapted from Cocoa/set_installation_options.sh shell script] 
      # keep the # symbol (i.e., leave these environmental keys unset in `set_installation_options.sh`)
      #export IGNORE_EMULTRF_CODE=1              # SaraivanovZhongZhu (SZZ) transformer/CNN-based emulators
      #export IGNORE_EMULTRF_DATA=1              # CMB, distance and power networks (Planck + SN + BAO and hybrid examples)
      #export IGNORE_PLANCK_LIKELIHOOD_CODE=1    # Planck + SN + BAO examples
      #export IGNORE_PLANCK_CMB_DATA=1           # Planck + SN + BAO examples
      #export IGNORE_SN_DATA=1                   # DES Y5 supernovae
      #export IGNORE_BAO_DATA=1                  # DESI DR2 BAO
      #export IGNORE_NAUTILUS_SAMPLER_CODE=1     # EXAMPLE_EMUL_NAUTILUS[1-2].py
      #export IGNORE_POLYCHORD_SAMPLER_CODE=1    # EXAMPLE_EMUL_POLY[1-2].yaml
      #export IGNORE_GETDIST_CODE=1              # Nautilus scripts and scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL*.py

> [!TIP]
> If one of these keys was active when `setup_cocoa.sh` ran, see the Cocoa
> [FAQ: How can users compile a single external module (not involving Cosmolike)?](https://github.com/CosmoLike/cocoa#appendix_compile_separately)

Now, users must follow all the steps below.

 **Step :one:**: Activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

 **Step :two:**: Ensure OpenMP is **OFF**.

    export OMP_NUM_THREADS=1

 **Step :three:** Run `cobaya-run` on the first emulator example following the commands below.

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by slot \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_EVALUATE1.yaml -f

  - macOS (arm)
 
        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_EVALUATE1.yaml -f
    
- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by slot \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC1.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC1.yaml -r

    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
          --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by slot \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC2.yaml -r

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_MCMC2.yaml -r
      
> [!Note]
> The examples below may require a large number of MPI workers. Before running them, it may be necessary to increase 
> the limit of threads that can be created (at UofA HPC type `ulimit -u 1000000`), otherwise users 
> may encounter the error `libgomp: Thread creation failed`

- **PolyChord**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY1.yaml -r

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY1.yaml -r
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY2.yaml -r

  - macOS (arm)
 
        mpirun -n 12 --oversubscribe \
          cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL_POLY2.yaml -r

For multi-node runs, see [MPI across nodes](#mpi_across_nodes).

> [!NOTE]
> **Note on core oversubscription**: an MPI process that is waiting still burns 100% of its
> core, checking for messages in a loop. With more processes than cores, this stalls the
> processes doing real work. Open MPI usually detects this and makes waiting processes give
> up the CPU, but its detection can be fooled. Adding `--mca mpi_yield_when_idle 1` forces
> that behavior; it is harmless otherwise.

The `Nautilus`, `Minimizer`, `Profile`, and `Emcee` scripts below contain an internally defined `yaml_string` that specifies priors, 
likelihoods, and the theory code, all following Cobaya Conventions.

- **Nautilus**:

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
            --root ./projects/lsst_y1/ --outroot "EXAMPLE_EMUL_NAUTILUS1"  \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_NAUTILUS1" \
            --maxfeval 750000 --nlive 2048 --neff 15000 \
            --flive 0.01 --nnetworks 5

  The Colab example [Test Nautilus](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSSTY1)_Test_Nautilus.ipynb) illustrates how stable Nautilus results are as a function of `nlive` 

  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux
    
        "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_NAUTILUS2"  \
            --maxfeval 850000 --nlive 3072 --neff 15000 \
            --flive 0.01 --nnetworks 5

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_EMUL_NAUTILUS2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_NAUTILUS2"  \
            --maxfeval 850000 --nlive 3072 --neff 15000 \
            --flive 0.01 --nnetworks 5
    
  What if the user runs a `Nautilus` chain with `--maxfeval` insufficient for producing `neff` samples? `Nautilus` saves the chain checkpoint at `chains/outroot_checkpoint.hdf5`.

- **Emcee**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 51 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE1" \
            --maxfeval 1000000

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE1" \
            --maxfeval 1000000
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 114 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE2" \
            --maxfeval 2000000

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_EMCEE2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_EMCEE2" \
            --maxfeval 2000000
    
  The number of steps per MPI worker is $n_{\rm sw} =  {\rm maxfeval}/n_{\rm w}$,
  with the number of walkers being $n_{\rm w}={\rm max}(3n_{\rm params},n_{\rm MPI})$.

  For proper convergence, each walker should traverse 50 times the autocorrelation length ($\tau$),
  which is provided in the header of the output chain file. A reasonable rule of thumb is to assume
  $\tau > 200$ and therefore set ${\rm maxfeval} > 10,000 \times n_{\rm w}$.
  Finally, the script sets burn-in (per walker) at $`5 \times \tau`$.

  With these numbers, users may ask when `Emcee` is preferable to `Metropolis-Hastings`?
  Here are a few numbers from the `Planck CMB (l < 396) + SN + BAO + LSST-Y1` test case.
  1) `MH` achieves convergence with $n_{\rm sw} \sim 150,000$ (number of steps per walker), but only requires four walkers.
  2) `Emcee` has $\tau \sim 300$, so it requires $n_{\rm sw} \sim 15,000$ when running with $n_{\rm w}=114$.
  
  Conclusion: `Emcee` requires $\sim 3$ more evaluations in this case, but the number of evaluations per MPI worker (assuming one MPI worker per walker) is reduced by $\sim 10$.
  Therefore, `Emcee` seems well-suited for chains where the evaluation of a single cosmology is time-consuming (and there is no slow/fast decomposition).

  What if the user runs an `Emcee` chain with `--maxfeval` insufficient for convergence? `Emcee` saves the chain checkpoint at `chains/outroot.h5`.

- **Sampler Comparison**

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_COMPARE_CHAINS_EMUL[2].py`.

  The Google Colab notebooks [Example Sampler Comparison (LSST-Y1 only)](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSSTY1).ipynb) and
  [Example Sampler Comparison (LSST+Others)](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSSTY1)_Sampler_Comparison_2.ipynb) can also reconstruct a similar version of these figures.

  <p align="center">
  <img width="750" height="750" alt="project_lsst_plot_sampler_comparison_1" src="https://github.com/user-attachments/assets/ffc72bb0-1843-4a55-9a69-ca4c7d6b34c2" />
  </p>

  Another Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` - $n_{\rm param} = 38$:

  <p align="center">
  <img width="750" height="750" alt="project_lsst_plot_sampler_comparison_2" src="https://github.com/user-attachments/assets/5bd7318e-864e-439d-9c3c-eaf07e267654" />
  </p>

- **Global Minimizer**:

  The minimizer reimplements `Procoli`, developed by Karwal et al (arXiv:2401.14225).

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 51 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_MIN1" \
            --nstw 350

  - macOS (arm)

        mpirun -n 12 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE1.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_MIN1" \
            --nstw 350
    
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 114 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE2.py \
            --root ./projects/lsst_y1/ \
            --outroot "EXAMPLE_EMUL_MIN2" \
            --nstw 750

  - macOS (arm)

        mpirun -n 12 \
          --oversubscribe python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE2.py \
          --root ./projects/lsst_y1/ \
          --outroot "EXAMPLE_EMUL_MIN2" \
          --nstw 750
    
  The number of steps per Emcee walker per temperature is $n_{\rm stw}$,
  and the number of walkers is $n_{\rm w}={\rm max}(3n_{\rm params},n_{\rm MPI})$.
  The minimum number of total evaluations is $3n_{\rm params} \times n_{\rm T} \times n_{\rm stw}$, which can be distributed among $n_{\rm MPI} = 3n_{\rm params}$ MPI processes for faster results.

  The scripts that generated the plots below are provided at `scripts/EXAMPLE_PLOT_MIN_COMPARE_CONV[2].py`

  <p align="center">
  <img width="750" height="750" alt="Minimum convergence against n_stw for LSST-Y1 cosmic shear (EXAMPLE_PLOT_MIN_COMPARE_CONV.py)" src="https://github.com/user-attachments/assets/31c36592-2d6c-4232-b5b4-5f686f9f2b8e" />
  </p>

  In the tests of these examples, $n_{\rm stw} \sim 200$ worked reasonably well up to $`n_{\rm param} \sim \mathcal{O}(10)`$.
  The case below, with $`n_{\rm param} = 38`$, illustrates the need for performing convergence tests on a case-by-case basis.

  In this example, the total number of evaluations for a reliable minimum is approximately $319,200$ ($n_{\rm stw} \sim 700$), distributed among $n_{\rm MPI} = 114$ processes for faster results.
  With the use of emulators, such minima can be computed with $\mathcal{O}(1)$ MPI workers.

  <p align="center">
  <img width="750" height="750" alt="Minimum convergence against n_stw for Planck + SN + BAO + LSST-Y1 (EXAMPLE_PLOT_MIN_COMPARE_CONV2.py)" src="https://github.com/user-attachments/assets/c43b8eea-ee2e-443d-a497-cb9b2dae2fc3" />
  </p>

- **Profile**: 

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 51 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE1.py \
            --root ./projects/lsst_y1/ --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 \
            --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN1.txt"

  - macOS (arm)
        
        mpirun -n 51 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE1.py \
            --root ./projects/lsst_y1/ \
            --cov 'chains/EXAMPLE_EMUL_MCMC1.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE1" \
            --factor 3 --nstw 350 --numpts 10 --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN1.txt"
       
  or (Example with `Planck CMB (l < 396) + SN + BAO + LSST-Y1` -  $n_{\rm param} = 38$)

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 114 --oversubscribe \
          -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x OMP_DYNAMIC \
          -x ROOTDIR -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
          -x CLIK_PLUGIN -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH \
          -x CLIK_DATA --mca mpi_yield_when_idle 1 --rank-by slot --map-by slot \
          --mca pml ob1 --mca btl vader,tcp,self --bind-to core:overload-allowed \
          --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 --report-bindings \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE2.py \
            --root ./projects/lsst_y1/ \
            --cov 'chains/EXAMPLE_EMUL_MCMC2.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE2" \
            --factor 3 --nstw 750 --numpts 10 --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2.txt"

  - macOS (arm)

        mpirun -n 114 --oversubscribe \
          python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE2.py \
            --root ./projects/lsst_y1/ \
            --cov 'chains/EXAMPLE_EMUL_MCMC2.covmat' \
            --outroot "EXAMPLE_EMUL_PROFILE2" \
            --factor 3 --nstw 750 --numpts 10 --profile 1 \
            --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2.txt"
     
  The argument `factor` specifies the start and end of the parameter being profiled:

      start value ~ minimum value - factor*np.sqrt(np.diag(cov))
      end   value ~ minimum value + factor*np.sqrt(np.diag(cov))

  Use ${\rm factor} \sim 3$ for parameters that are well constrained by the data when a covariance matrix is provided.
  If `cov` is not supplied, the code estimates one internally from the prior.
  If a parameter is poorly constrained or `cov` is not given, use $`{\rm factor} \ll 1`$.

  The script of the plot below is provided at `projects/lsst_y1/scripts/EXAMPLE_PLOT_PROFILE1[2].py`

  Profile 1: `LSST-Y1 Cosmic Shear only`

  The Google Colab [Profile Likelihood (LSST-Y1 only)](https://github.com/CosmoLike/CoCoAGoogleColabExamples/blob/main/Cocoa_Example_(LSST_Y1)_Profile_Likelihoods.ipynb) can be used to reconstruct a similar figure.
  
  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_profile1" src="https://github.com/user-attachments/assets/11f0f0dd-23e6-4875-bd8e-afbb11ac4e48" />
  </p>

  Profile 2: `Planck CMB (l < 396) + SN + BAO + LSST-Y1 Cosmic Shear`

  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_profile2" src="https://github.com/user-attachments/assets/cd041f96-dc42-426e-84a7-2d6498218b5f" />
  </p>

# Running Hybrid Cosmolike-ML emulators <a name="cobaya_base_code_examples_emul2"></a>

> [!NOTE]
> These hybrid examples remain experimental. The checks below verify the
> workflow; assess emulator accuracy and posterior convergence for your analysis.

The `EXAMPLE_EMUL2` examples emulate the background expansion and matter
power spectra. CosmoLike still computes the survey projections, bias and
intrinsic-alignment contributions. Changing n(z) or nuisance parameters does
not require retraining a survey data-vector network.

The shared theory networks live in `external_modules/data/emultrf`. Install
them through the [main Cocoa emulator recipe](https://github.com/CosmoLike/cocoa#cobaya_base_code_examples_emul2).
These networks assume **mnu = 0.06 eV**; do not sample neutrino mass. Their
cold-matter power approximation is not a calibrated massive-neutrino halo
model. Check their training range before widening cosmological priors.

We assume Cocoa and this project are installed, the Cocoa Conda environment
is active, the shell is Bash, and the current folder is `cocoa/Cocoa/`.

**Step :one:**: activate Cocoa.

```bash
source start_cocoa.sh
```

**Step :two:**: Select the number of OpenMP cores (below, we set it to 4).

  - Linux

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=4; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :three:**: remove GPU access on Linux; these examples use the CPU.

```bash
export CUDA_VISIBLE_DEVICES=""
```

**Step :four:**: evaluate the first hybrid example.

- Linux

      "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe \
        --mca pml ob1 --mca btl vader,tcp,self \
        --bind-to core:overload-allowed --report-bindings \
        --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
        cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_EVALUATE1.yaml --force

- macOS (arm)

      mpirun -n 1 --oversubscribe \
        cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_EVALUATE1.yaml --force


The YAML selects the CPU for the distance emulator. Keep BLAS at one thread
per MPI rank (`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`); on macOS also
use `VECLIB_MAXIMUM_THREADS=1`. The Python sampler entry points set these
BLAS limits before importing numerical libraries.

| Example | Configuration 1 | Configuration 2 |
|---|---|---|
| Fixed evaluation | [EXAMPLE_EMUL2_EVALUATE1.yaml](EXAMPLE_EMUL2_EVALUATE1.yaml) | [EXAMPLE_EMUL2_EVALUATE2.yaml](EXAMPLE_EMUL2_EVALUATE2.yaml) |
| Cobaya MCMC | [EXAMPLE_EMUL2_MCMC1.yaml](EXAMPLE_EMUL2_MCMC1.yaml) | [EXAMPLE_EMUL2_MCMC3.yaml](EXAMPLE_EMUL2_MCMC3.yaml) |
| Annealed minimization | [EXAMPLE_EMUL2_MINIMIZE1.py](EXAMPLE_EMUL2_MINIMIZE1.py) | [EXAMPLE_EMUL2_MINIMIZE2.py](EXAMPLE_EMUL2_MINIMIZE2.py) |
| Parameter profile | [EXAMPLE_EMUL2_PROFILE1.py](EXAMPLE_EMUL2_PROFILE1.py) | [EXAMPLE_EMUL2_PROFILE2.py](EXAMPLE_EMUL2_PROFILE2.py) |
| Nautilus sampling | [EXAMPLE_EMUL2_NAUTILUS1.py](EXAMPLE_EMUL2_NAUTILUS1.py) | [EXAMPLE_EMUL2_NAUTILUS2.py](EXAMPLE_EMUL2_NAUTILUS2.py) |

Configuration **1** uses `lsst_y1.cosmic_shear`, NLA, and `lsst_y1_M1_GGL0.05.dataset`.
Configuration **2** uses `lsst_y1.combo_3x2pt`, NLA, and `lsst_y1_M1_GGL0.05.dataset`.
[EXAMPLE_EMUL2_MCMC2.yaml](EXAMPLE_EMUL2_MCMC2.yaml) adds Planck 2018 CMB
(`l < 396` plik TTTEEE, low-ℓ TT and EE), DESI DR2 BAO and DES Y5 supernovae
to configuration 1: the `Planck CMB (l < 396) + SN + BAO + LSST-Y1`
combination of the emulator examples.

The minimization, profile and Nautilus scripts read the corresponding
`EXAMPLE_EMUL2_EVALUATE1.yaml` or `2.yaml`; `--input` selects another evaluate
YAML. They require `cocoa_hybrid_sampling.py` from the matching shared core
revision. They do not maintain separate embedded cosmologies. `--check` evaluates
the specified fiducial and prints the sampled parameter order without sampling.
Use a new `--outroot` for each run; these scripts refuse to overwrite results.

### Cobaya MCMC

With Cocoa activated in Bash from `cocoa/Cocoa/`, run the first MCMC
example. Use
`EXAMPLE_EMUL2_MCMC3.yaml` for configuration 2 and `EXAMPLE_EMUL2_MCMC2.yaml`
for the Planck + BAO + SN combination. Check chain convergence before
interpreting posterior constraints.

**Step :one:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :two:**: remove GPU access on Linux; these examples use the CPU.

```bash
export CUDA_VISIBLE_DEVICES=""
```

**Step :three:**: start Cobaya's hybrid MCMC.

- Linux

      "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe \
        --mca pml ob1 --mca btl vader,tcp,self \
        --bind-to core:overload-allowed --report-bindings \
        --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
        cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_MCMC1.yaml

- macOS (arm)

      mpirun -n 4 --oversubscribe \
        cobaya-run ./projects/lsst_y1/EXAMPLE_EMUL2_MCMC1.yaml


### Minimization, profiles and Nautilus

We assume Cocoa and this project are installed, the Cocoa Conda environment
is active, the shell is Bash, and the current folder is `cocoa/Cocoa/`.

**Step :one:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :two:**: remove GPU access on Linux; these examples use the CPU.

```bash
export CUDA_VISIBLE_DEVICES=""
```

**Step :three:**: check the hybrid setup before a long run.

```bash
python ./projects/lsst_y1/EXAMPLE_EMUL2_MINIMIZE1.py --check
```

**Step :four:**: search for a minimum.

- Linux

      "${CONDA_PREFIX}"/bin/mpirun -n 21 --oversubscribe \
        --mca pml ob1 --mca btl vader,tcp,self \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
        -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
        -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
        -x CLIK_PLUGIN -x CUDA_VISIBLE_DEVICES --mca mpi_yield_when_idle 1 \
        --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
        --bind-to core:overload-allowed --report-bindings \
        --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
        python ./projects/lsst_y1/EXAMPLE_EMUL2_MINIMIZE1.py --nstw 200 --outroot hybrid_min1

- macOS (arm)

      mpirun -n 12 --oversubscribe \
        python ./projects/lsst_y1/EXAMPLE_EMUL2_MINIMIZE1.py --nstw 200 --outroot hybrid_min1


**Step :five:**: profile the first sampled parameter using that saved minimum.

- Linux

      "${CONDA_PREFIX}"/bin/mpirun -n 21 --oversubscribe \
        --mca pml ob1 --mca btl vader,tcp,self \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
        -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
        -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
        -x CLIK_PLUGIN -x CUDA_VISIBLE_DEVICES --mca mpi_yield_when_idle 1 \
        --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
        --bind-to core:overload-allowed --report-bindings \
        --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
        python ./projects/lsst_y1/EXAMPLE_EMUL2_PROFILE1.py --profile 0 --nstw 200 --numpts 11 --factor 1 --minfile ./projects/lsst_y1/chains/hybrid_min1.json --outroot hybrid_profile1

- macOS (arm)

      mpirun -n 12 --oversubscribe \
        python ./projects/lsst_y1/EXAMPLE_EMUL2_PROFILE1.py --profile 0 --nstw 200 --numpts 11 --factor 1 --minfile ./projects/lsst_y1/chains/hybrid_min1.json --outroot hybrid_profile1


**Step :six:**: run Nautilus as an independent sampling example.

- Linux

      "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
        --mca pml ob1 --mca btl vader,tcp,self \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
        -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
        -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
        -x CLIK_PLUGIN -x CUDA_VISIBLE_DEVICES --mca mpi_yield_when_idle 1 \
        --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
        --bind-to core:overload-allowed --report-bindings \
        --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
        python ./projects/lsst_y1/EXAMPLE_EMUL2_NAUTILUS1.py --nlive 1000 --neff 10000 --maxfeval 100000 --outroot hybrid_nautilus1

- macOS (arm)

      mpirun -n 12 --oversubscribe \
        python ./projects/lsst_y1/EXAMPLE_EMUL2_NAUTILUS1.py --nlive 1000 --neff 10000 --maxfeval 100000 --outroot hybrid_nautilus1


The annealed Emcee search follows the DES × Planck template. Its objective
is **−2 log posterior**, including nuisance and cosmological priors; the
profile is therefore a penalized profile, not a pure likelihood profile.
`--nstw` sets steps per walker per temperature. More steps and independent
starts are needed to assess whether a minimum is reliable.

`--profile` accepts a sampled-parameter name or its printed zero-based index.
`--factor` gives the half-width in proposal standard deviations, clipped to
the prior bounds. `--cov` accepts a covariance whose header lists the sampled
parameters in order; without it, the prior covariance sets the proposal.
The minimum JSON must come from the same evaluate YAML and parameter order.
Older plain-text minimum files are not accepted. Set any additional priors
in the input YAML; these scripts do not insert hidden cosmological priors.

Nautilus writes weighted GetDist-compatible rows and a JSON convergence
record. Reaching `--maxfeval` is not convergence. If the budget ends before
any posterior samples are retained, only the checkpoint and a JSON record
with `converged: false` are saved. Its prior transform uses
Cobaya's one-dimensional prior distributions; external prior factors enter
once as additional log weight. Evidence with unnormalized external priors
has that normalization limitation. These examples do not certify emulator
accuracy or posterior convergence.

### Emulator design and optional approximations

Details on the matter power spectrum emulator designs will be presented in the [emulator_code](https://github.com/SBU-COSMOLIKE/emulators_code) repository.

Neural networks generalize the *syren-new* Eq. 6 of [arXiv:2410.14623](https://arxiv.org/abs/2410.14623) formula for the linear power spectrum (w0waCDM with a fixed neutrino mass of $0.06$ eV) to new models, extended ranges, or higher precision. Similar networks generalize the *syren-Halofit* LCDM nonlinear boost fit (Eq. 11 of [arXiv:2402.17492](https://arxiv.org/abs/2402.17492)).

> [!NOTE]
> Users can decide not to correct the *syren-new* formula for the linear power spectrum (flag in the yaml). The caveats of the syren-new approximation have not been studied extensively; it appears sufficient for w0waCDM forecasts when combined with the Euclid Emulator to compute the nonlinear boost.
>
> For back-of-the-envelope LCDM calculations (e.g., to test cosmolike features), users can also choose not to correct the *syren-Halofit* formula for the LCDM nonlinear boost (see figure below). In this case, the overhead on top of cosmolike computations is minimum, at the order of $0.01$ seconds on a macOS M2Pro laptop.
>
> <p align="center">
>  <span style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">
>    <img width="450" alt="compare_emul_hemul" src="https://github.com/user-attachments/assets/7ada9b3b-db01-499f-8170-a2db5ef90636" />
>    <img width="450" alt="Second figure of the syren-Halofit comparison" src="https://github.com/user-attachments/assets/d19f0900-cebc-41ca-9028-1d4e0bd40cc9" />
>  </span>
> </p>


### MPI across nodes <a name="mpi_across_nodes"></a>

> [!NOTE]
> **Running on more than one node.**
>
> 1. **Network interface.** TCP must use an interface routable between compute
>    nodes. A common exclusion list is
>    `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0`; adapt it to the cluster.
>    Keep `ib0` if routable IP-over-InfiniBand is the intended network. These
>    examples exchange parameter vectors and scalar scores, so communication
>    volume is small; actual scaling still depends on the machine.
> 2. **Environment.** Remote ranks need the same Cocoa paths and libraries:
>    `ROOTDIR`, `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, OpenMP
>    and BLAS settings, and `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN` when used.
>    Slurm normally exports the submitting environment (`--export=ALL`).
>    Explicit `-x` options also forward these variables with SSH launchers.
>    Activate Cocoa before launching; build/download flags do not replace
>    runtime paths. All nodes must see the same files at the same paths.
> 3. **Slurm geometry.** Keep `ntasks-per-node × cpus-per-task` within the
>    allocated physical cores per node. Set `OMP_NUM_THREADS` to
>    `SLURM_CPUS_PER_TASK` and use `--map-by numa:pe=${OMP_NUM_THREADS}`.
>    The minimization, profile and Nautilus pool reserves one MPI rank as
>    coordinator; the remaining ranks evaluate the model.

Within a Slurm allocation, first activate Cocoa in Bash on the launch node.
The following steps assume Open MPI 4 and shared installation/data paths.
Omit optional CLIK exports if those variables are not set.

**Step :one:**: match OpenMP threads to the scheduler allocation.

    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"; export OMP_PROC_BIND=close; \
    export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
    export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :two:**: launch the hybrid minimizer across the allocated ranks.

```bash
"${CONDA_PREFIX}"/bin/mpirun -n "${SLURM_NTASKS}" \
  --mca pml ob1 --mca btl vader,tcp,self \
  --mca btl_tcp_if_exclude lo,docker0,virbr0 \
  --map-by numa:pe=${OMP_NUM_THREADS} --bind-to core --report-bindings \
  -x ROOTDIR -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CUDA_VISIBLE_DEVICES \
  python ./projects/lsst_y1/EXAMPLE_EMUL2_MINIMIZE1.py --nstw 200 --outroot hybrid_multinode
```

For a Planck likelihood add `-x CLIK_PATH -x CLIK_DATA -x CLIK_PLUGIN` when
those variables are defined. Follow the cluster's MPI module and Slurm
launch policy; do not oversubscribe a production allocation. Outside Slurm,
supply the hosts and slots with the cluster's `--hostfile` or `--host` recipe.

# Running Fisher <a name="lsst_examples_fisher"></a>

  Section 18 of the Jupyter notebook `projects/lsst_y1/EXAMPLE_EVALUATE1.ipynb` computes cosmic-shear Fisher matrices for 17 parameters (five cosmological, two NLA, five source photo-z shifts and five shear calibrations) with five-point-stencil derivatives. It tests how they depend on `AccuracyBoost` and on the step size, and repeats the `AccuracyBoost` test with DerivativeKit derivatives. Gaussian priors on the photo-z shifts and shear calibrations enter the Fisher matrix; hard (flat) priors only clip the GetDist contours.

  <p align="center">
  <img width="1156" height="858" alt="example_lssty1_fisher" src="https://github.com/user-attachments/assets/4f4e5775-6522-4eca-af82-4d1a10f4b1c8" />
  </p>

  Section 18 of `projects/lsst_y1/EXAMPLE_EVALUATE2.ipynb` computes the 3×2pt Fisher matrix of 29 parameters (seven cosmological plus the NLA, photo-z, shear-calibration and lens-bias nuisances) at the example's own accuracy settings, and checks the prior cuts of the example yamls against the forecast; the [accuracy FAQ](#accuracy_choice) quotes its numbers.



# Unit tests <a name="unit_tests"></a>

The `tests/` folder compares likelihood predictions against stored reference
values, checks for race conditions from OpenMP threading, and measures the
numerical error of the default accuracy settings. It also checks the
accuracy of the hybrid emulated pipelines.
[tests/README.md](tests/README.md) describes the checks, how to run them,
the input data snapshots, and how to refresh them.

# Exploring notebooks <a name="notebooks"></a>

**Armadillo** was chosen to make a convenient Python API for notebook
exploration. This C++ library provides vectors, matrices and three-dimensional
arrays called cubes. A small interface layer connects them to NumPy through
**pybind11**, with **CARMA** handling array conversion. The notebooks expose
intermediate quantities; production calculations use the CLI interfaces.

We assume Cocoa and this project are installed, the Cocoa Conda environment
is active, the shell is Bash, and the current folder is `cocoa/Cocoa/`.

Compile the project first; the covariance notebook also needs the optional
covariance build described [below](#computing_covariances).

**Step :one:**: activate Cocoa.

```bash
source start_cocoa.sh
```

**Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :three:**: start Jupyter.

```bash
jupyter notebook --no-browser --port=8888
```

**Step :four:**: open the printed URL and choose a notebook below.

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

| Notebook | Contents |
|---|---|
| [EXAMPLE_EVALUATE1.ipynb](EXAMPLE_EVALUATE1.ipynb) | Cosmic shear: $`C_\ell^{EE}`$ and $`\xi_\pm`$, angular rebinning, feedback from hydrodynamical simulations (`baryon_sims`), CAMB and CosmoLike accuracy, $\chi^2$ against the stored data vector, Halofit versus EuclidEmulator2, the wavenumbers each measurement probes, and Fisher forecasts. |
| [EXAMPLE_EVALUATE2.ipynb](EXAMPLE_EVALUATE2.ipynb) | Galaxy–galaxy lensing ($`C_\ell^{gs}`$, $`\gamma_t`$) and clustering (Limber and non-Limber $`C_\ell^{gg}`$, $`w(\theta)`$): rebinning, simulation-based feedback, numerical accuracy, the 3×2pt $`\chi^2`$, and Halofit versus EuclidEmulator2. |
| [EXAMPLE_EVALUATE3.ipynb](EXAMPLE_EVALUATE3.ipynb) | The `bfmt` feedback models (three SP(k) relations, BCEmu, Flamingo, BCemu2025) applied to the 3×2pt data vector, their $\chi^2$ against the feedback-free data, and one-at-a-time parameter sweeps of each model's suppression; needs `bfmt` installed ([baryonic feedback](#lsst_y1_baryonic_feedback)). |
| [EXAMPLE_EVALUATE_COVARIANCE.ipynb](EXAMPLE_EVALUATE_COVARIANCE.ipynb) | G, SSC, cNG, total, separate 1h–4h matter trispectra and matrix diagnostics; needs the covariance build. |

Each notebook opens with its model-and-data table and a "Before running" note
on requirements and run time; read them before running all cells. Choose the
Python kernel from the activated Cocoa environment and restart it after
recompiling. The [covariance guide](covariance/README.md) explains the
forecast files, figures and refinement workflow.


# Computing covariances <a name="computing_covariances"></a>

The covariance tools compute Gaussian (G), super-sample (SSC), connected
non-Gaussian (cNG) and total matrices before scale cuts.
[covariance/README.md](covariance/README.md) explains compilation, the production
CLI, exploratory notebooks, accuracy settings and output files.

# Appendix <a name="appendix"></a>

## FAQ: Which accuracy settings are available? <a name="accuracy"></a>

Data-vector options belong to the selected `likelihood` block. Covariance
options belong to the evaluate YAML's `covariance` block. They use separate
names and settings; changing one does not refine the other.

| Data-vector setting | What it changes |
|---|---|
| `accuracyboost` | Overall interpolation-table resolution. |
| `integration_accuracy` | Quadrature resolution; levels 5/6/7 allow 2048/4096/8192 points per Limber panel. Refine independently of interpolation. |
| `internal_accuracyboost` | C-FAST-PT convolution grid. |
| `nonlimber_accuracyboost` | Non-Limber distance sampling. |
| `pk_z_refinement` | Nested redshift refinement of matter-power inputs. |
| `lmax` | Real-space angular-transform cutoff, where a real-space transform is used. |
| `kmax_boltzmann` | Requested Boltzmann power range; coordinate it with the theory settings. |

`adopt_limber_gg` and `adopt_limber_gs` choose a projection approximation.
`photoz_interpolation_type` chooses how n(z) is interpolated, while
`photoz_zmid_convention` describes the input coordinates. These are modeling
or input-convention choices, not interchangeable accuracy boosts.

`photoz_histogram_akima: 0` retains the original midpoint n(z) reconstruction; `1` selects histogram preparation followed by Akima for source and lens samples. `photoz_histogram_akima_factor` chooses 3 (default), 5 or 7 preparation cells per original uniform bin. See the [reconstruction comparisons and area plots](tests/README.md#optional_studies). The selected first factor never changes with an accuracy boost.


The default data-vector power grid has 1,500 wavenumbers at boost 1.
Covariance alone uses `power_accuracyboost: 8` to prepare 11,993 nodes by
natural cubic interpolation before C linear lookup. Its `accuracy_boost`
refines tables and cutoffs; its `integration_accuracy` independently selects
quadrature levels 0–4. See the complete [covariance accuracy table](covariance/README.md#accuracy-settings).

CAMB's `theory.camb.extra_args.AccuracyBoost` controls CAMB, not CosmoLike.
Check interpolation, quadrature, input-power sampling and transform cutoffs
separately at fixed cosmology and measurement bins. Narrow n(z) overlaps
particularly require a quadrature check; increasing `accuracyboost` alone
is not that check. The [data-vector test guide](tests/data_vector/README.md)
and [covariance test guide](tests/covariance/README.md) state what each set of tests
actually verifies. A passing regression or a larger boost is not a general
claim of survey or Fisher convergence.

## FAQ: Which accuracy settings should I choose? <a name="accuracy_choice"></a>

**The defaults hold near the fiducial cosmology.** The likelihood
configurations (`likelihood/*.yaml`) ship `accuracyboost: 1`,
`integration_accuracy: 0` and matching defaults for the other settings of
the [previous FAQ](#accuracy). These values keep the numerical error of
the data vector small near the fiducial cosmology. The N-random-models
check measures this: it compares the default settings with every accuracy
setting pushed far beyond them, at random points of the prior. When
$`\Omega_m`$, $`A_s`$ and the lens biases are held at the fiducial and
only the other parameters move, the defaults keep almost every model
below the 0.2 budget; the exceptions sit at the edge of the
intrinsic-alignment prior, very far from anything data-like, and are
explained below.
The synthetic data vectors of the tests sit at the fiducial by
construction, so this is the regime the tests certify.

**Why the error grows away from the fiducial.** Let the finite accuracy
settings leave a roughly fixed fractional error $`\varepsilon`$ on the
theory vector $`m`$, so the data-vector error is
$`\delta = \varepsilon\, m`$. An MCMC keeps the covariance $`C`$ fixed
at the fiducial model, so

```math
\Delta\chi^2 = \delta^{\rm T} C^{-1} \delta
             = \varepsilon^2\, m^{\rm T} C^{-1} m
```

grows with the signal while the "ruler" $`C`$ does not. At linear order
the 3×2pt pieces scale with the primordial amplitude $`A_s`$ (sampled as
`As_1e9` $`= 10^9 A_s`$) and the linear galaxy bias $`b_i`$ of lens bin
$`i`$ (`LSST_B1_i`):

| Piece | Signal scales as | Factor on $`\Delta\chi^2`$ |
|---|---|---|
| $`w(\theta)`$, galaxy clustering of lens bin $`i`$ | $`A_s b_i^2`$ | $`[(A_s/A_s^{\rm fid})(b_i/b_i^{\rm fid})^2]^2`$ |
| $`\gamma_t(\theta)`$, galaxy–galaxy lensing of lens bin $`i`$ | $`A_s b_i`$ | $`[(A_s/A_s^{\rm fid})(b_i/b_i^{\rm fid})]^2`$ |
| $`\xi_\pm(\theta)`$, cosmic shear | $`A_s`$ | about $`(A_s/A_s^{\rm fid})^2`$ |

The shear factor is "about" because the nonlinear power grows faster than
linearly in $`A_s`$, and $`\Omega_m`$ enters the lensing kernels as well.
The real-space statistics $`w(\theta)`$ and $`\gamma_t(\theta)`$
inherit these scalings from the Fourier-space power spectra they
transform.

These factors hold whatever the noise level, because only the numerator
of $`\Delta\chi^2`$ changes; the noise sets how much each band weighs at
the fiducial. For one Gaussian band of a power spectrum $`C_\ell`$
measured with $`N_{\rm modes}`$ modes and noise $`N`$ (shot noise for
clustering, shape noise for shear), the band error is

```math
\sigma(C_\ell) \propto \sqrt{2/N_{\rm modes}}\, (C_\ell + N).
```

- Signal-dominated bands ($`C_\ell \gg N`$):
  $`\sigma_{\rm fid} \approx C_\ell^{\rm fid}\sqrt{2/N_{\rm modes}}`$,
  so the band contributes about $`(N_{\rm modes}/2)\,\varepsilon^2`$
  times the factor above.
- Noise-dominated bands ($`C_\ell \ll N`$):
  $`\sigma_{\rm fid} \approx N\sqrt{2/N_{\rm modes}}`$, so the
  contribution is suppressed by $`(C_\ell^{\rm fid}/N)^2`$. It still
  grows with the same factor.

**Why real data call for more caution.** With simulated data the best fit
sits at the fiducial by construction, so the defaults are enough. With
real data you don't know beforehand how far the best fit and the
posterior will be from the fiducial model the covariance was built at.
Accuracy then has to hold over that wider region, where the amplitude
factors above inflate the error.

A second reason is the residual term. The tests measure
$`\Delta\chi^2 = \delta^{\rm T} C^{-1} \delta`$ against noiseless
synthetic data, where the residual $`r = m - d`$ between the theory
$`m`$ and the data $`d`$ is zero. On real data the numerical error
changes $`\chi^2`$ by

```math
\Delta\chi^2 = 2\, \delta^{\rm T} C^{-1} r
             + \delta^{\rm T} C^{-1} \delta .
```

The first term is first order in $`\delta`$ and weighted by the
residual. Real data can hold pieces that are a bit of an outlier; the
Planck TT and EE spectra have several multipoles like that. Where $`r`$
is large, the same numerical error moves $`\chi^2`$ much more than the
tests show.

The residual term is linear in the same error $`\delta`$, and by the
Cauchy–Schwarz inequality

```math
\lvert 2\, \delta^{\rm T} C^{-1} r \rvert
  \leq 2 \sqrt{\delta^{\rm T} C^{-1} \delta}\,
          \sqrt{r^{\rm T} C^{-1} r} .
```

So keeping $`\delta^{\rm T} C^{-1} \delta`$ small over a wider range
of $`\Omega_m`$ and $`A_s b_i^2`$, which is what the N-random-models
check tests, also bounds the outlier-driven term. Even without a test on
real residuals, the pipeline gains that robustness as a by-product.

**For MCMC on real data, set `integration_accuracy: 3`.** Levels 3
through 10 agree with each other to about $`\Delta\chi^2 = 0.01`$ at
every point tested, so level 3 is converged.

The table gives $`\Delta\chi^2`$ against the pushed accuracy settings,
on the 3×2pt data vector, measured 2026-11-05. "Held" means $`\Omega_m`$,
$`A_s`$ and the lens biases at the fiducial.

| Point | $`\Delta\chi^2`$, level 0 | $`\Delta\chi^2`$, level 3 |
|---|---:|---:|
| $`\Omega_m = 0.26`$, amplitude ratio $`(A_s/A_s^{\rm fid})(b_i/b_i^{\rm fid})^2 = 3.1`$ | 1.99 | 0.08 |
| $`\Omega_m = 0.35`$, ratio 3.5 | 1.67 | 0.008 |
| held (ratio 1), $`H_0 = 89.5`$ km/s/Mpc, $`w_0 + w_a = -2.63`$ | 0.38 | 0.056 |
| $`\Omega_m = 0.46`$, ratio 4.2 | 1.19 | 0.22 |

At the fiducial, levels 0 and 3 differ by $`\Delta\chi^2 = 0.0013`$. A
full evaluation takes about 2.7 s at level 3 against 2.6 s at level 0
(laptop, one data vector), so the cost is nearly the same.

**The defaults are not safe at the edge of the prior.** The
intrinsic-alignment amplitude makes this concrete. NLA scales the
alignment signal with the redshift power law

```math
A_{\rm IA}(z) = A_1 \left( \frac{1+z}{1.62} \right)^{\eta} ,
```

where $`A_1`$ (`LSST_A1_1`) and $`\eta`$ (`LSST_A1_2`) are each drawn
flat in $`[-5, 5]`$ (`likelihood/params_source.yaml`), and the fiducial
point is $`A_1 = 0.606`$, $`\eta = -1.515`$. The held model with the
worst error at level 0 has $`\eta = 4.65`$ and $`A_1 = 3.59`$, with
$`\Omega_m`$, $`A_s`$ and the biases at the fiducial; another bad point
has $`\eta = 4.43`$. A redshift power $`\eta \approx 4`$ is very far
from anything data-like. A chain that must be precise there needs
`integration_accuracy: 3` or higher, for the following reason.

**Aliasing of the n(z) bumps.** The intrinsic-alignment terms of the
shear $`C_\ell`$ (GI, alignment times lensing, and II, alignment times
alignment) integrate the source n(z) directly. That n(z) is a cubic
spline through the 300 histogram cells of the source file, so it
carries small bumps on the cell scale ($`\Delta z \approx 0.012`$) and
sharp edges where each tomographic bin's shifted support ends. The
Limber integral uses a fixed Gauss–Legendre rule whose node count
follows `integration_accuracy`: 96, 128, 256 and 512 nodes at levels 0,
1, 2 and 3.

With too few nodes (the spacing at 96 nodes is about 0.036 in z, three
cells), each node lands at an arbitrary phase of the bumps. The
weighted sum does not cancel them, and a leftover remains whose size
and sign depend only on where the nodes fall. That is aliasing: the
same effect that makes a car wheel seem to spin backwards on video.
The error therefore does not shrink steadily with the node count; it
jumps around. Measured for the highest-redshift bin pair at the
$`\eta = 4.65`$ point, the relative error of the II integral is
$`-1.3\times10^{-3}`$ at 96 nodes, $`+5.2\times10^{-3}`$ at 128 nodes,
$`+4.6\times10^{-4}`$ at 256 nodes and $`+4.1\times10^{-5}`$ at 512
nodes. From 512 nodes on (level 3) the bumps are resolved, and levels 3
to 10 agree.

The leftover is tiny on its own. The factor $`((1+z)/1.62)^{\eta}`$
with $`\eta`$ between about +4 and +5 multiplies the high-redshift end
of the integrand, where n(z) is noisiest, by about $`10^4`$ to $`10^5`$
relative to the fiducial. That turns the leftover into $`\Delta\chi^2`$
of order 10 to 100. With the alignment amplitudes set to zero, every
level agrees to $`\Delta\chi^2 = 7\times10^{-5}`$. Over 99.9% of the
error sits in cosmic shear $`\xi_\pm`$.

The example yamls also carry prior cuts that keep the sampler out of
corners where no data set would place the posterior:

- $`\Omega_m`$ in $`[0.2, 0.5]`$;
- $`w_0 + w_a \geq -3`$;
- the amplitude cut $`(A_s/A_s^{\rm fid})(b_i/b_i^{\rm fid})^2 \leq 8`$
  on every lens bin, a cobaya `prior` block.

Such corners are numerically costly because $`\Delta\chi^2`$ grows with
the amplitude factors above.

Section 18 of [EXAMPLE_EVALUATE2.ipynb](EXAMPLE_EVALUATE2.ipynb) checks
these cuts against the 3×2pt Fisher forecast of the example, which gives
$`\sigma(\Omega_m) = 0.024`$, $`\sigma(w_0) = 0.27`$,
$`\sigma(w_a) = 0.83`$ and $`{\rm FoM}(w_0, w_a) = 17.3`$. The
$`2\sigma`$ contours reach an amplitude ratio of only 1.29–1.32 in every
lens bin, $`\Omega_m`$ spans $`[0.24, 0.36]`$ against the box
$`[0.2, 0.5]`$, and $`w_0 + w_a`$ spans $`[-2.32, 0.52]`$, clear of
$`-3`$: every numerical cut lies far outside what the data allow. Only
the physical wall at $`w_0 + w_a < 0`$, which keeps dark energy
subdominant at early times, crosses the $`2\sigma`$ region.

A one-setting-at-a-time scan at the worst points found that the other
settings did not matter: `accuracyboost: 2`, the non-Limber grid and
CAMB's wavenumber range moved $`\Delta\chi^2`$ by at most 0.17. CAMB's
`AccuracyBoost: 2` moved it by up to 1.15, and only at the extreme corner
($`\Omega_m = 0.49`$, ratio 4.7), which the cuts and the posterior do
not reach.

The tests themselves run `integration_accuracy: 3`: the frozen references
pin it through `FROZEN_LIKELIHOOD_OVERRIDES` in
[tests/cocoa_test_utils.py](tests/cocoa_test_utils.py), while the example
yamls ship level 0.

See [tests/data_vector/README.md](tests/data_vector/README.md),
subsection "The N-random-models check (opt-in)", for the full
measurements and figures.
