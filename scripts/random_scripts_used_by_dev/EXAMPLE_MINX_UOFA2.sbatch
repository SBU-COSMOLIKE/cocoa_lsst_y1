#!/bin/bash

#SBATCH --job-name=RY1_MIN1
#SBATCH --output=RY1_MIN1-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --ntasks-per-socket=14
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=3
#SBATCH --time=50:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --exclusive

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
conda --version

cd $SLURM_SUBMIT_DIR
conda activate cocoapy38
source start_cocoa

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/EXAMPLE_MIN${SLURM_ARRAY_TASK_ID}.yaml -f