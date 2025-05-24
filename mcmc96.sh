#!/bin/bash

#SBATCH --job-name=mcmc_96
#SBATCH --output=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/projects/lsst_y1/logs/MCMC%a_run_%A.out
#SBATCH --error=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/projects/lsst_y1/logs/MCMC%a_run_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=7
#SBATCH --partition=extended-96core
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu
#SBATCH -t 7-00:00:00

export x1=$(($SLURM_ARRAY_TASK_ID+1))
export x2=$(($SLURM_ARRAY_TASK_ID+2))
yaml_file1=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/projects/lsst_y1/yamls/MCMC${SLURM_ARRAY_TASK_ID}.yaml
yaml_file2=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/projects/lsst_y1/yamls/MCMC${x1}.yaml
yaml_file3=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/projects/lsst_y1/yamls/MCMC${x2}.yaml

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
module purge > /dev/null 2>&1

source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
cd /gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa
module load slurm
conda activate cocoa
source start_cocoa.sh

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

$CONDA_PREFIX/bin/mpirun -n 4 --oversubscribe --report-bindings --mca vader,btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run $yaml_file1 -r & $CONDA_PREFIX/bin/mpirun -n 4 --oversubscribe --report-bindings --mca vader,btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run $yaml_file2 -r & $CONDA_PREFIX/bin/mpirun -n 4 --oversubscribe --report-bindings --mca vader,btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run $yaml_file3 -r