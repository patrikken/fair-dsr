#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --array=0-7
#SBATCH --ntasks=83
#SBATCH --mem-per-cpu=16G 
 
module load python/3.10
module load mpi4py/3.1.3

base_models=(lr rf gbm)
threshold_uncert=(0.1 0.4 0.3 0.2 0.5 0.6 0.62 0.64 0.65 0.68 0.7)
fair_metrics=(dp eodds eop)
demographic_predictors=(DNN OURS) 

 
for fair_metric in "${fair_metrics[@]}" 
do	    
    for thresh in "${threshold_uncert[@]}"
    do
        srun python3 ./models/Ablation_Proxy_Evaluation.py --treshold_uncert=$thresh --fair_metric=$fair_metric --seed=$SLURM_ARRAY_TASK_ID --is_adv_method  --dataset $1
        for base_model in "${base_models[@]}"
        do
            srun python3 ./models/Ablation_Proxy_Evaluation.py --treshold_uncert=$thresh --base_model=$base_model --fair_metric=$fair_metric --seed=$SLURM_ARRAY_TASK_ID  --dataset $1
        done
    done
done 
