#!/bin/bash
#SBATCH --job-name=unisim_sample
#SBATCH --error=logs/unisim_%A_%a.err
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --partition=long
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-91

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Index from the Slurm array
idx="${SLURM_ARRAY_TASK_ID}"

python infer_prot.py \
    --config ./config/infer_prot.yaml \
    --index $idx \
    --max_iter_energy_minimization 10 \
    --energy_eval_budget 10_000

python infer_prot.py \
    --config ./config/infer_prot.yaml \
    --index $idx \
    --max_iter_energy_minimization 100 \
    --energy_eval_budget 10_000

python infer_prot.py \
    --config ./config/infer_prot.yaml \
    --index $idx \
    --max_iter_energy_minimization 1_000 \
    --energy_eval_budget 10_000

python infer_prot.py \
    --config ./config/infer_prot.yaml \
    --index $idx \
    --max_iter_energy_minimization 100 \
    --energy_eval_budget 1_000_000

python infer_prot.py \
    --config ./config/infer_prot.yaml \
    --index $idx \
    --max_iter_energy_minimization 1_000 \
    --energy_eval_budget 1_000_000
