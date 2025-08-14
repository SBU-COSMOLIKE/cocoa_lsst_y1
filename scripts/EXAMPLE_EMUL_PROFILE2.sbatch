#!/bin/bash

#SBATCH --job-name=PROF2
#SBATCH --output=PROF2-%A-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=94
#SBATCH --ntasks-per-node=94
#SBATCH --ntasks-per-socket=47
#SBATCH --cpus-per-task=1
#SBATCH --time=9:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

sleep $(( 10 + SLURM_ARRAY_TASK_ID*20 )) # avoid different scripts running start_cocoa.sh at the same time

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate cocoa
source start_cocoa.sh

export OMP_NUM_THREADS=1
ulimit -u 2000000 # require line when nmpi is high

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --rank-by slot \
  --map-by slot:pe=${OMP_NUM_THREADS} --mca mpi_yield_when_idle 1 \
  python ./projects/lsst_y1/EXAMPLE_EMUL_PROFILE2.py \
    --root ./projects/lsst_y1/ --cov 'chains/EXAMPLE_EMUL_MCMC2.covmat' \
    --outroot "EXAMPLE_EMUL_PROFILE2" --factor 3 --nstw 700 --numpts 10 \
    --profile ${SLURM_ARRAY_TASK_ID} \
    --minfile="./projects/lsst_y1/chains/EXAMPLE_EMUL_MIN2.txt"