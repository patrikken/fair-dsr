
# Fairness Under Demographic Scarce Regime

This repository contains the code for the paper [Fairness Under Demographic Scarce Regime](https://arxiv.org/abs/2307.13081). Demographic Scarce Regime (DSR) refers to settings where demographic information (sensitive attribute) is not *fully* available. The paper studies the properties of the sensitive attribute classifier that can affect the fairness-accuracy tradeoffs of the downstream classifier. It draws a link between the uncertainty of the sensitive attribute prediction and the fairness-accuracy tradeoff obtained w.r.t to the true sensitive attributes.


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

other dependencies a located in [requiments.txt](./README.md) file

## Experiments on fair classification with different sensitive attributes baselines

### Data 

Download each dataset and store them  (preprocessed) in the folder [preprocessing](./preprocessing/). For each dataset create two different csv files for each subsets: $\mathcal{D}_1$ (dataset without sensitive attributes) and $\mathcal{D}_2$ (with sensitive attributes) as described in the paper. The file [src/datasets.py](src/datasets.py) contains code to load each dataset separated in subsets $\mathcal{D}_1$ and $\mathcal{D}_2$


### Sensitive attribute classifier with uncertainty awarenes
The file `src/demographic_predictor.py` contains code to train the attribute classifier with uncertainty estimation. 

#### Train fair model with predicted sensitive attributes 
The file `src/Proxy_Evaluation.py` contains code to train and evaluate fair models with different attribute classifier baselines (proxies).

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
* `dataset` (string): Specify the dataset to be used:  `adult`, `compas_race`, `new_adult`, `celeba` 
* `seed` (int): Id of the sample. Choose between 0 and 7
* `fair_metric` (`dp`, `eodds`, `eop`): Fairness metric 
  - `dp`: Demographic Parity
  - `eop`: Equal Opportunity
  - `eodds`: Equalized Odds
* `demographic_predictor` (DNN, KNN): Model used to infer the sensitive attribute from the related feature. This parameter is used when `sensitive_feature_type` is `predicted`. Possible values `DNN` and `KNN`. 
  - `DNN` use MLP based attribute classifier
  - `KNN` use KNN based attribute classifier (imputation)
* `is_adv_method` (boolean): Use adversarial debaising method if True. Else use Reduction methods. 
* `base_model` (string): Base classifier for reduction methods. 
  - `lr`: LogisticRegression
  - `rf`: RandomForest
  - `gbm`: GradientBoostingClassifier
* `sensitive_feature_type` (string): Use clean or predicted sensitive features. 
  - `clean`: apply fairness mechanism w.r.t  ground truth sensitive attribute  
  - `ours`: apply fairness mechanism w.r.t mostly certain predicted sensitive attributes
  - `predicted`: apply fairness mechanism w.r.t MLP or KNN based attribute classifier. 


## Other baselines with fairness constraints
Use the file [src/target_predictor.py](src/target_predictor.py) to train the target classifier with fairness mechanisms not supported by fairlearn. 

#### Parameters:

* `baseline` (ARL, DRO, FAIR_BATCH, VANILLA): specify the baseline to use to train the target classifier. 
  - `ARL`: train the classifier with Adversarially Reweighted Learning (ARL) by Lahoti et al. (2020.). 
  - `DRO`:  train the classifier with [robust loss](https://github.com/daniellevy/fast-dro);  distributionally robust optimization (DRO) by Hashimoto et al. (2018). 
  - `CVAR`:  train the classifier with robust loss and KL-regularized (fast DRO) by Levy et al. (2020).
  - `FAIR_BATCH`: train the classifier with [Fairbatch](https://github.com/yuji-roh/fairbatch/blob/main/FairBatchSampler.py) by Roh, Yuji, et al. 
  - `VANILLA`: train the classifier without fairness constraints. 
* `dataset` (string): specify the dataset to be used:  `adult`, `compas_race`, `new_adult`, `celeba`. 


## Baselines without fairness constraints
Use file [src/Unfair_Evaluation.py](src/Unfair_Evaluation.py) to run the baselines without fairness constraints. It uses the parameters `--base_model`, `--dataset`, and `--seed` as described above. 

## Reproducing the analysis in the paper
We assume that you have done all the experiments for every baselines, the results for each baseline and each seed are stored in the folder `output`. 
To aggregate the results across seeds, run the file [analysis/compute_average.py](analysis/compute_average.py) with the argument `--dataset` specifying the dataset.


#### Creating the plots. 
The notebook [analysis/plots.ipynb](analysis/plots.ipynb) has functions to plot the results in the paper. The plots are saved in the folder `analysis/results/plots`. 

## Experiments on different uncertainty thresholds 
For this experiment, use the file [src/Ablation_Proxy_Evaluation.py](src/Ablation_Proxy_Evaluation.py) to train fair classifier for different uncertainty thresholds by setting the parameter `--treshold_uncert` to define the confidence threshold. The results for each baseline and for each seed are stored in the folder `output/{dataset}/ablation` and for each dataset specified with the parameter `--dataset` as mentioned above.  



## Citation 

```
@article{kenfack2023fairness,
  title={Fairness Under Demographic Scarce Regime},
  author={Kenfack, Patrik Joslin and Kahou, Samira Ebrahimi and A{\"\i}vodji, Ulrich},
  journal={arXiv preprint arXiv:2307.13081},
  year={2023}
}
```