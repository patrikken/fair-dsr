#!/bin/bash
#SBATCH --time=40:00:00 
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus-per-task=1 
#SBATCH --cpus-per-task=2 

module load python/3.10 

python3 models/demographic_predictor.py --accelerator gpu 