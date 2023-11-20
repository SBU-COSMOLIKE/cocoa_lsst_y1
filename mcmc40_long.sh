#!/bin/bash

#SBATCH --job-name=mcmc_emu
#SBATCH --output=/gpfs/projects/MirandaGroup/jonathan/cocoa/Cocoa/projects/lsst_y1/chains/MCMC%a/run_%A.out
#SBATCH --error=/gpfs/projects/MirandaGroup/jonathan/cocoa/Cocoa/projects/lsst_y1/chains/MCMC%a/run_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --partition=long-40core
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jonathan.gordon@stonybrook.edu
#SBATCH -t 2-00:00:00

yaml_file=/gpfs/projects/MirandaGroup/jonathan/cocoa/Cocoa/projects/lsst_y1/yamls/MCMC${SLURM_ARRAY_TASK_ID}.yaml
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
module purge > /dev/null 2>&1

source /gpfs/home/jsgordon/miniconda/etc/profile.d/conda.sh
module load slurm
conda activate cocoapy38
source start_cocoa

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --report-bindings --mca vader,btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run $yaml_file -r