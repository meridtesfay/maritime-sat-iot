#!/bin/bash
#SBATCH --job-name=maritime-sat-iot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Setup
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load modules (adjust for your HPC)
module purge
module load cuda/11.8
module load python/3.9

# Activate environment
source activate maritime

# Create logs directory
mkdir -p logs checkpoints

# Train all models
echo "Training CNN model..."
python training/train.py --model cnn --epochs 50 --batch_size 64

echo "Training LSTM model..."
python training/train.py --model lstm --epochs 50 --batch_size 64

echo "Training Transformer model..."
python training/train.py --model transformer --epochs 50 --batch_size 64

echo "Training Hybrid model..."
python training/train.py --model hybrid --epochs 50 --batch_size 64

echo "All models trained!"
echo "Job finished at: $(date)"
