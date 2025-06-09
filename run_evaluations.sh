#!/bin/bash
#SBATCH --job-name=eval    # Job name
#SBATCH --output=logs/eval/pythia_%j.out        # Standard output and error log (%j expands to jobID)
#SBATCH --error=logs/eval/pythia_%j.err         # Error log
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --account=OPEN-34-14
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --gpus=1                      # Number of GPUs requested
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=64GB                    # Memory limit
#SBATCH --partition=qgpu               # Partition name

#!/bin/bash
cd /mnt/raid/data/Petr/NeuralCDCL
source activate main_env

# Run immediately first
echo "$(date): Starting initial evaluation" 
CUDA_VISIBLE_DEVICES=1 python inference.py inference.modelpath=temp/hf_Pythia-12-16-256-up inference.datapath=data/generalization/up/
echo "$(date): Initial evaluation finished"

# Then loop every 2 hours
while true; do
    echo "$(date): Sleeping for 2 hours..."
    sleep 7200
    echo "$(date): Starting scheduled evaluation" 
    CUDA_VISIBLE_DEVICES=1 python inference.py inference.modelpath=temp/hf_Pythia-12-16-256-up inference.datapath=data/generalization/up/
    echo "$(date): Scheduled evaluation finished"
done
