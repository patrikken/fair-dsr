#!/bin/bash

sbatch hpc_scripts/method_clean.sh $1
sbatch hpc_scripts/method_ours.sh $1
sbatch hpc_scripts/method_predictor.sh $1