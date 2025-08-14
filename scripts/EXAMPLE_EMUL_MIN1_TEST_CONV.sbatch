#!/bin/bash

#SBATCH --job-name=MIN2T
#SBATCH --output=MIN2T-%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=52
#SBATCH --ntasks-per-node=52
#SBATCH --ntasks-per-socket=26
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
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

sleep $(( RANDOM%61 )) # avoid different scripts running start_cocoa.sh at the same time

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate cocoa
source start_cocoa.sh

export OMP_NUM_THREADS=1
ulimit -u 2000000 # require line when nmpi is high

for ((j=0;j<10;j++)); do
  declare -i nstw=$((100+25*j))
  mpirun -n ${SLURM_NTASKS} --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self \
    --bind-to core:overload-allowed --rank-by slot \
    --map-by slot:pe=${OMP_NUM_THREADS} --mca mpi_yield_when_idle 1 \
    python ./projects/lsst_y1/EXAMPLE_EMUL_MINIMIZE1.py --root ./projects/lsst_y1/ \
      --outroot "EXAMPLE_EMUL_MIN_TEST_CONV${j}" --nstw ${nstw}
done