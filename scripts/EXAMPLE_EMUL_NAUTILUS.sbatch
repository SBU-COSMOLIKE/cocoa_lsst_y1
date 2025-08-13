#!/bin/bash

#SBATCH --job-name=ENAUL
#SBATCH --output=ENAUL-%A-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=90
#SBATCH --ntasks-per-node=90
#SBATCH --ntasks-per-socket=45 
#SBATCH --cpus-per-task=1  
#SBATCH --time=24:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --exclusive

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

mpirun -n 90 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --rank-by slot \
    --bind-to core:overload-allowed --map-by slot --mca mpi_yield_when_idle 1 \
    python -m mpi4py.futures ./projects/example/EXAMPLE_EMUL_NAUTILUS${SLURM_ARRAY_TASK_ID}.py \
        --root ./projects/lsst_y1/ --outroot "EXAMPLE_EMUL_NAUTILUS${SLURM_ARRAY_TASK_ID}"  \
        --maxfeval 750000 --nlive 2048 --neff 15000 --flive 0.01 --nnetworks 5