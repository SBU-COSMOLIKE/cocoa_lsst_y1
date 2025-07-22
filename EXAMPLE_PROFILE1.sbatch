#!/bin/bash
#SBATCH --job-name=PRF1
#SBATCH --output=PRF1-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --ntasks-per-node=80
#SBATCH --ntasks-per-socket=40
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --partition=standard
#SBATCH --account=cosmolike
#SBATCH --exclusive

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

ulimit -u 2000000 # require line when nmpi is high
export nmpi=80
export numpts=79

mpirun -n ${nmpi} --oversubscribe --mca pml ^ucx  \
  --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS}  \
  python -m mpi4py.futures ./projects/lsst_y1/EXAMPLE_PROFILE1.py \
  --cov 'EXAMPLE_MCMC1.covmat' --nwalkers 5 --profile ${SLURM_ARRAY_TASK_ID} \
  --maxfeval 9000 --numpts ${numpts} --factor 10