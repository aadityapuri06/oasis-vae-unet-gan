#!/bin/bash

# --job-name: shown in squeue; student ID included per course convention.
#SBATCH --job-name=s4895678_unet_train

# --partition / --account: comp3710 course partition and billing account,
# same as the existing ResNet18 job.
#SBATCH --partition=comp3710
#SBATCH --account=comp3710

# --gres: request an A100 specifically, matching the existing ResNet18 job.
#SBATCH --gres=gpu:a100:1

# --time: wall-clock limit. Locally (CPU only, no cudnn benchmarking or
# mixed precision) the smoke test ran ~5 batches/epoch in ~30-40s, i.e.
# roughly 6-8s/batch. Scaled to the full training set (~9664 images at
# batch_size=64, ~151 batches/epoch), that's ~151 * 7s =~ 18 min/epoch on
# CPU, or ~9 hours for all 30 epochs. An A100 should speed the actual
# conv/backward compute up a lot, but num_workers=0 means data loading
# stays single-threaded and could limit how much of that speedup is
# realized, and evaluate_full_dice() also does a full pass over the
# validation set every epoch on top of training. 3 hours gives generous
# margin over the realistic GPU estimate without assuming the best case.
#SBATCH --time=03:00:00

# --mem: host RAM. Model and batches are small; this comfortably covers
# the DataLoader and cached PIL/numpy arrays for the full dataset.
#SBATCH --mem=16G

# --cpus-per-task: CPU cores for the job. train_unet.py uses num_workers=0
# (unchanged from the notebook), so this is headroom for the main
# process's own data loading/decoding rather than parallel workers.
#SBATCH --cpus-per-task=4

# --output / --error: SLURM stdout/stderr logs, one per job (%j = job ID).
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Sanity-check the dataset location before spending GPU time on a bad
# path - the exact folder name/layout under /home/groups/comp3710 may
# differ slightly from the local zip's structure, so print what's
# actually there. train_unet.py also fails fast with a clear error if
# data_root doesn't exist, but this gives more detail in the log to fix
# the path with if it's wrong.
echo "--- Listing /home/groups/comp3710 ---"
ls -la /home/groups/comp3710 2>&1
echo "--- Listing /home/groups/comp3710/OASIS ---"
ls -la /home/groups/comp3710/OASIS 2>&1
echo "--- Listing /home/groups/comp3710/OASIS/keras_png_slices_data ---"
ls -la /home/groups/comp3710/OASIS/keras_png_slices_data 2>&1

# Load any modules required by the cluster environment (e.g. CUDA, Python).
# Uncomment/adjust these to match what's available on Rangpur.
# module load python/3.10
# module load cuda/12.1

# Activate a virtual environment if you're using one. Adjust the path to
# wherever your venv lives.
# source venv/bin/activate

python train_unet.py

echo "End time: $(date)"
