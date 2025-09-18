# CT2Rep
MICCAI 2024 & CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging
 
## debug 



## Requirements

Before you start, you will need to install the necessary dependencies. To do so, execute the following commands:

```setup
# Navigate to the 'ctvit' directory and install the required packages
cd ctvit
pip install -e .

# Return to the root directory
cd ..
```

## Dataset

An example dataset is provided [example_data_ct2rep.zip](https://huggingface.co/generatect/GenerateCT/blob/main/example_data_ct2rep.zip). This is to show the required dataset structure for CT2Rep and CT2RepLong. For the full dataset, please see [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

Good instruction. 
https://vlm3dchallenge.com/getting-started/


## Train

To train the models, go to the corresponding directory, and run the command

```train
python main.py --max_seq_length 300 --threshold 10 --epochs 100 --save_dir results/test_ct2rep/ --step_size 1 --gamma 0.8 --batch_size 1 --d_vf 512
```
The threshold is the minimum number of instances in the dataset for a token to be put in the tokens dictionary. You can select the directories, xlsx files, longitudinal files, etc. with the corresponding keyword arguments.



## full job 
```bash
#!/bin/bash
#SBATCH --job-name=quick_train_classification
#SBATCH --time=01:00:00
#SBATCH --output=logs-het/het.log
#SBATCH --error=logs-het/het-err.log
#SBATCH --distribution=cyclic

# Heterogeneous job component 1: A cluster
#SBATCH --nodes=2
#SBATCH --partition=A
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=public&1a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64

# Heterogeneous job component 2: BB cluster
#SBATCH hetjob
#SBATCH --partition=BB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=csce&4a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64

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
```