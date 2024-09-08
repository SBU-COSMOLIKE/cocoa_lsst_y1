#!/bin/bash

#SBATCH --job-name=EP5
#SBATCH --output=EP5-%A-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=13
#SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=7
#SBATCH --time=72:00:00
#SBATCH --partition=standard
#SBATCH --account=cosmolike
#SBATCH --exclusive

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
conda activate cocoa
source start_cocoa.sh

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

cd $SLURM_SUBMIT_DIR/projects/lsst_y1

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} \
  python -m mpi4py.futures EXAMPLE_PROFILE1.py --AB 1.0 --tol 0.02 --profile 1 \
  --maxfeval 1000 --mpi 12 --outroot "monday" --minmethod 5