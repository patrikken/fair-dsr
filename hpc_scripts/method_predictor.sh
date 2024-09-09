#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --array=0-7
#SBATCH --ntasks=83
#SBATCH --mem-per-cpu=8G

module load python/3.10
module load mpi4py/3.1.3

base_models=(lr rf gbm)
sensitive_ft_types=(predicted clean)
fair_metrics=(dp eodds eop)
demographic_predictors=(DNN KNN) 
method=predicted # other values ours, clean
 
for fair_metric in "${fair_metrics[@]}" 
do	    
    for dem_predictor in "${demographic_predictors[@]}"
        do
            srun python3 ./models/Proxy_Evaluation.py --demographic_predictor=$dem_predictor --sensitive_feature_type=predicted --fair_metric=$fair_metric --seed=$SLURM_ARRAY_TASK_ID --is_adv_method  --dataset $1
            for base_model in "${base_models[@]}"
                do
                    srun python3 ./models/Proxy_Evaluation.py --demographic_predictor=$dem_predictor --base_model=$base_model --sensitive_feature_type=predicted --fair_metric=$fair_metric --seed=$SLURM_ARRAY_TASK_ID  --dataset $1
                done
        done
    done 
done 
