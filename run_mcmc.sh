#!/bin/bash
#SBATCH --job-name=COLA
#SBATCH --time=7-00:00
#SBATCH --output=./projects/lsst_y1/logs/%x_%a_%A.out
#SBATCH --error=./projects/lsst_y1/logs/%x_%a_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mail-user=joao.reboucas@unesp.br
#SBATCH --mail-type=ALL

YAML=./projects/lsst_y1/yamls/MCMC${SLURM_ARRAY_TASK_ID}.yaml

echo "Job started in `hostname`"

cd ~/cocoa/Cocoa
conda activate cocoa
source start_cocoa.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close

mpirun -n ${SLURM_NTASKS_PER_NODE} --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by core --map-by core cobaya-run ${YAML} -r
