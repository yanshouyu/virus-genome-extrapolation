#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job script: NT-v2 MLM domain adaptation on Baculoviridae genomes
#
# Adjust the directives below for your cluster (partition, account, time limit).
# Submit from the project root:
#   sbatch scripts/run_pretrain.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=nt-domain-adapt
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --account=def-aucoin
#SBATCH --mail-user=shouyu.yan@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Environment ────────Uncomment and set──────────────────────────────────────
source ~/projects/def-aucoin/shouyu/venvs/huggingface/bin/activate

# ── Working directory ─────────────────────────────────────────────────────────
[ -d logs ] || mkdir -p logs

echo "──────────────────────────────────────────"
echo "Job        : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "GPU(s)     : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "Started    : $(date)"
echo "──────────────────────────────────────────"

# ── Training ──────────────────────────────────────────────────────────────────
export HF_HUB_DISABLE_XET=1

python -m nt_domain_adapt.pretrain \
    --base-model  "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species" \
    --gb-dir      "data/raw/baculovirus/Baculoviridae" \
    --window-nt   6144 \
    --stride-nt   3072 \
    --mlm-probability 0.15 \
    --epochs      10 \
    --lr          1e-4 \
    --effective-batch-size 128 \
    --early-stopping-patience 3

deactivate
echo "──────────────────────────────────────────"
echo "Finished   : $(date)"
echo "──────────────────────────────────────────"
