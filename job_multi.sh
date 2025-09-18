#!/bin/bash
#SBATCH --job-name=CT2Rep
#SBATCH --time=3-00:00:00
#SBATCH --output=logs_multi/%j_%N.out
#SBATCH --error=logs_multi/%j_%N.err
#SBATCH --distribution=cyclic

# Heterogeneous job component 1: A cluster
#SBATCH --nodes=4
#SBATCH --partition=vgpu
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=public&1v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32

# Heterogeneous job component 2: BB cluster
#SBATCH hetjob
#SBATCH --partition=vgpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=public&1v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32


# From now on, unless you use more than one cluster or more complex threading/processes, you don't need to change anything below except the sbatch flags above and the script training path. 
SCRIPT_TRAINING=train_apptainer.sh 

echo "Heterogeneous job leader (component 0) starting on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
TOTAL_NODES=$(env | awk -F= '/^SLURM_JOB_NUM_NODES_HET_GROUP_/ {sum += $2} END {print (sum ? sum : 0)}')

# technically you can for loop here, but I prefer to roll out by hand right now because the goal is to keep it easy to catch up as much concept about SLURM environment as possible, not bash script nor python. 
SLURM_JOB_NUM_NODES_HET_GROUP_Th=0
srun --het-group=0 --nodes=$SLURM_JOB_NUM_NODES_HET_GROUP_0 --ntasks-per-node=1 bash -c "bash $SCRIPT_TRAINING $TOTAL_NODES $(hostname) $SLURM_JOB_NUM_NODES_HET_GROUP_Th" &
srun --het-group=1 --nodes=$SLURM_JOB_NUM_NODES_HET_GROUP_1 --ntasks-per-node=1 bash -c "bash $SCRIPT_TRAINING $TOTAL_NODES $(hostname) $SLURM_JOB_NUM_NODES_HET_GROUP_0" &

wait
echo "Done printing hostnames."