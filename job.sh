#!/bin/bash
#SBATCH --job-name=up_pythia    # Job name
#SBATCH --output=logs/train/pythia_%j.out        # Standard output and error log (%j expands to jobID)
#SBATCH --error=logs/train/pythia_%j.err         # Error log
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --account=OPEN-34-14
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --gpus=1                      # Number of GPUs requested
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --mem=64GB                    # Memory limit
#SBATCH --partition=qgpu               # Partition name

module load Anaconda3/2024.02-1
module load CUDA/12.4.0

source activate main_env

python train_pythia.py