from argparse import ArgumentParser, Namespace
import os
import pandas as pd
import numpy as np
from ndf import is_pareto_efficient


parser = ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="adult", help="dataset")
parser.add_argument("--model_name", type=str,
                    default="all", help="model_name")
arg = parser.parse_args()


base_models = ['lr', 'rf', 'gbm']
sensitive_ft_types = ["clean", "predicted", "ours"]
fair_metrics = ["dp", "eop", "eodds"]
demographic_predictors = ["DNN", "KNN"]

results_path = "./outputs/{}/results/averaged/".format(arg.dataset)
pareto_path = "./analysis/results/{}/ablation/".format(arg.dataset) 
base_dir = "./outputs/{}/ablation/".format(arg.dataset)
 

os.makedirs(results_path, exist_ok=True)
os.makedirs(base_dir, exist_ok=True)


def average_results(output_dir, file_name, seeds):
    if len(seeds) == 0:
        return
    to_ret = {"acc": [], "unfair": [], "std_unfair": [], "std_acc": []}
    for file in seeds:
        df = pd.read_csv(file)
        if df.shape[0] > 2:
            # only consider runs for all epsilons
            df.sort_values('epsilon')
            to_ret["acc"].append(df.acc.values)
            to_ret["unfair"].append(df.unfair.values)
            eps = df["epsilon"].values
    accuracies = np.array(to_ret["acc"]).mean(axis=0)
    unfairness = np.array(to_ret["unfair"]).mean(axis=0)
    std_unfair = np.array(to_ret["unfair"]).std(axis=0)
    std_acc = np.array(to_ret["acc"]).std(axis=0)

    print(accuracies.shape, unfairness.shape)
    to_ret["acc"] = accuracies
    to_ret["unfair"] = unfairness
    to_ret["std_unfair"] = std_unfair
    to_ret["std_acc"] = std_acc
    to_ret["epsilon"] = eps 
 

 
    output_file = "{}{}.csv".format(output_dir, file_name) 

    errors = 1.0 - accuracies
    pareto_input = [[err, unfair] for (err, unfair) in zip(errors, unfairness)]
    msk = is_pareto_efficient(np.array(pareto_input))
    print(len(msk))
    df = pd.DataFrame()
    df['accuracy'] = [1.0 - errors[i] for i in range(len(errors)) if msk[i]]
    df['unfairness'] = [unfairness[i] for i in range(len(errors)) if msk[i]]
    df['std_unfair'] = [std_unfair[i] for i in range(len(errors)) if msk[i]]
    df['std_acc'] = [std_acc[i] for i in range(len(errors)) if msk[i]]
    df['epsilon'] = [eps[i] for i in range(len(errors)) if msk[i]] 
    df.to_csv(output_file, encoding='utf-8', index=False)


def get_result_sensitive_attr(base_dir, is_adv_debaising=False, sentive_attr_type="clean"):
    file_names = []
    results = {} 

    if is_adv_debaising:
        for fair_metric in fair_metrics:
            file_name = "avd_debaising_model_{}_ours".format(
                fair_metric)
            file_names.append(file_name)
    else:
        for model in base_models:
            for fair_metric in fair_metrics:
                file_name = "{}_model_{}_ours".format(model, fair_metric)
                file_names.append(file_name)

    for file_name in file_names:
        results[file_name] = []
        for r in range(7):
            file_path = "{}{}_{}.csv".format(base_dir, file_name, r) 
            if os.path.isfile(file_path): 
                results[file_name].append(file_path)
    return results


files = {}

values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for val in values:
    folder = "{}ours_{}/".format(base_dir, val)
    output_dir = "{}ours_{}/".format(pareto_path, val)
    os.makedirs(output_dir, exist_ok=True)

    clean = (get_result_sensitive_attr(base_dir=folder,
                                       is_adv_debaising=False))
     
    files.update(clean)
    files.update(get_result_sensitive_attr(base_dir=folder,
                                           is_adv_debaising=True))

    for file in files:
        average_results(output_dir, file, files[file])
