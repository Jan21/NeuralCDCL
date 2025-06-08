#!/bin/bash
cd /mnt/raid/data/Hyner_Petr/NeuralCDCL
source /mnt/raid/miniconda3/etc/profile.d/conda.sh
conda activate ph_rl_sos

# Run immediately first
echo "$(date): Starting initial evaluation" 
CUDA_VISIBLE_DEVICES=1 python inference.py inference.modelpath=temp/hf_Pythia-12-16-256-ca inference.datapath=data/generalization/ac/
echo "$(date): Initial evaluation finished"

# Then loop every 2 hours
while true; do
    echo "$(date): Sleeping for 2 hours..."
    sleep 7200
    echo "$(date): Starting scheduled evaluation" 
    CUDA_VISIBLE_DEVICES=1 python inference.py inference.modelpath=temp/hf_Pythia-12-16-256-ca inference.datapath=data/generalization/ac/
    echo "$(date): Scheduled evaluation finished"
done
