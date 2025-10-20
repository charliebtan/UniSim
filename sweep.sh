#!/bin/bash
#SBATCH --job-name=unisim_sweep
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --partition=unkillable,main,long
#SBATCH --array=0-80
#SBATCH --output=logs/unisim_%A_%a.out
#SBATCH --error=logs/unisim_%A_%a.err

INDEX=$SLURM_ARRAY_TASK_ID

which python

python infer_prot.py \
    --index $INDEX \
    --config ./config/infer_prot.yaml
