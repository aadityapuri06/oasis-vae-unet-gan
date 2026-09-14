#!/bin/bash
#SBATCH --job-name=s4895678_gan_train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"

python gan.py

echo "End time: $(date)"