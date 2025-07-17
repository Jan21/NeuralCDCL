#!/bin/bash
#SBATCH --job-name=eval_ca
#SBATCH --output=logs_eval/eval/ac/ac_%j.out
#SBATCH --error=logs_eval/eval/ac/ac_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=OPEN-34-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --partition=qgpu

module load Anaconda3/2024.02-1
module load CUDA/12.4.0
source activate main_env

python utils/inference.py inference.modelpath=temp/hf_Pythia-12-16-256-ac-debug-cos inference.datapath=data/generalization/ac/