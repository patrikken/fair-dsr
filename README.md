
# Fairness Under Demographic Scarce Regime

This repository contains the code to reproduce the experiments in our paper.


## Requirements
The project requires the following python packages:

* numpy
* pandas
* h5py
* mpi4py
* scikit-learn 
* tensorflow 
* folktables
* fairlearn 
* tensorboard
* torchvision

other packages a located in [requiments.txt](./README.md) file

## Experiments on fair classification with different sensitive attributes baselines

### Data preprocessing

The preprocessed data are located in the [preprocessing](./preprocessing/) folder 


### Sensitive attribute classifier
Use `demographic_predictor.py` to train the attribute classifier for each data. 

### Train fair model with predicted sensitive attributes 
Use `Proxy_Evaluation.py` to train fair models with different attributes classifier baselines.

#### On your local machine. 
Assuming `nbr_core` is the number of core you want to use:
```
cd core
mpiexec -n nbr_core python Proxy_Evaluation.py
```

#### On a HPC Cluster
You will have to provide the number of cores in your submission file and srun will use all the core available:
```
cd core
srun python Proxy_Evaluation.py
```

#### Parameters:
* `dataset` (string): Specify the dataset to be used:  `adult`, `compas_race`, `new_adult` 
* `seed` (int): Id of the sample. Choose between 0 and 7
* `fair_metric` (string): Fairness metric `dp`: Demographic Parity; `eop`:Equal Opportunity; `eodds`: Equalized Odds
* `demographic_predictor` (string): Model used to infer the sensitive attribute from the related feature. `DNN`, `KNN`
* `is_adv_method` (boolean): Use adversarial method if True. Else use Reduction methods
* `base_model` (string): Base classifier for reduction methods, `lr`:LosgisticRegression, `rf`:RandomForest, `gbm`:GradientBoostingClassifier
* `sensitive_feature_type` (string): Use clean or predicted sensitive features. `clean`, `ours` or `predicted`. 


## Reproducing the analysis in the paper
We assume that you have done all the experiments for every baselines, the results for each baseline and each seed are stored in folder `output`. 
To aggregate the results run the file [analysis/compute_average.py](analysis/compute_average.py) with specified `--dataset` as argument.

## Baselines without fairness constraints
Use file [src/Unfair_Evaluation.py](src/Unfair_Evaluation.py) to train the baselines without fairness constraints. It uses the parameters `--base_model`, `--dataset`, and `--seed` as described above. 

#### Creating the graphs. 
Results are saved in analysis/results/plots. The notebook [analysis/plots.ipynb](analysis/plots.ipynb) has functions to plot the results in the paper.

## Experiments on different uncertainty thresholds 
For this experiment, use the file [src/Ablation_Proxy_Evaluation.py](src/Ablation_Proxy_Evaluation.py) to run our method for different uncertainty threshold using parameter `--treshold_uncert` to set the threshold. The results for each baseline and each seed are stored in folder `output/{dataset}/ablation` for each dataset specified with the parameter `--dataset` as mentioned above.  
 