#!/bin/bash

# Setup
cd /mnt/raid/data/Hyner_Petr/NeuralCDCL
source /mnt/raid/miniconda3/etc/profile.d/conda.sh
conda activate ph_rl_sos
mkdir -p logs

# Run immediately first
echo "$(date): Starting initial evaluation" 
CUDA_VISIBLE_DEVICES=1 python utils/inference.py inference.modelpath=temp/hf_Pythia-12-16-256-ca inference.datapath=data/generalization/ac/
echo "$(date): Initial evaluation finished"

# Then loop every 1 hour
while true; do
    echo "$(date): Sleeping for 1 hour..."
    sleep 3600
    echo "$(date): Starting scheduled evaluation" 
    CUDA_VISIBLE_DEVICES=1 python utils/inference.py inference.modelpath=temp/hf_Pythia-12-16-256-ca inference.datapath=data/generalization/ac/
    echo "$(date): Scheduled evaluation finished"
done