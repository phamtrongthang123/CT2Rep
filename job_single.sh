#!/bin/bash
#SBATCH --job-name=CT2Rep
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j_%N.out
#SBATCH --error=logs/%j_%N.err
#SBATCH --distribution=cyclic

#SBATCH --nodes=2
#SBATCH --partition=agpu72
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16


# From now on, unless you use more than one cluster or more complex threading/processes, you don't need to change anything below except the sbatch flags above and the script training path. 
SCRIPT_TRAINING=train_apptainer.sh 

echo "Heterogeneous job leader (component 0) starting on $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
TOTAL_NODES=$(env | awk -F= '/^SLURM_JOB_NUM_NODES_HET_GROUP_/ {sum += $2} END {print (sum ? sum : 0)}')
# Fallback to regular SLURM_JOB_NUM_NODES if no heterogeneous groups found
if [ "$TOTAL_NODES" -eq 0 ]; then
    TOTAL_NODES=${SLURM_JOB_NUM_NODES:-1}
    SLURM_JOB_NUM_NODES_HET_GROUP_0=$TOTAL_NODES
fi
echo "TOTAL_NODES=$TOTAL_NODES"

# technically you can for loop here, but I prefer to roll out by hand right now because the goal is to keep it easy to catch up as much concept about SLURM environment as possible, not bash script nor python. 
SLURM_JOB_NUM_NODES_HET_GROUP_Th=0
srun --nodes=$SLURM_JOB_NUM_NODES_HET_GROUP_0 --ntasks-per-node=1 bash -c "bash $SCRIPT_TRAINING $TOTAL_NODES $(hostname) $SLURM_JOB_NUM_NODES_HET_GROUP_Th" &

wait
echo "Done printing hostnames."