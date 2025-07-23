#!/bin/bash

#SBATCH --job-name=MIN1
#SBATCH --output=MIN1-%A.out
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
conda activate cocoa
source start_cocoa

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

ulimit -u 2000000 # require line when nmpi is high
export nmpi=${SLURM_NTASKS}
export nwalkers=$((${SLURM_NTASKS}-1))

mpirun -n 5 --oversubscribe --mca pml ^ucx  \
  --mca btl vader,tcp,self --bind-to core:overload-allowed \
  --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS}  \
  python ./projects/lsst_y1/EXAMPLE_MINIMIZE1.py --root ./projects/lsst_y1/ \
  --cov 'EXAMPLE_EMUL_MCMC1.covmat' --outroot "example_min1" --nwalkers 5 --maxfeval 10000