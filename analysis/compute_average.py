from argparse import ArgumentParser, Namespace
import os
import pandas as pd
import numpy as np
from ndf import is_pareto_efficient


parser = ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="new_adult", help="dataset")
parser.add_argument("--model_name", type=str,
                    default="all", help="model_name")
arg = parser.parse_args()


base_models = ['lr', 'rf', 'gbm']
sensitive_ft_types = ["clean", "predicted", "ours", "ours_w"]
fair_metrics = ["dp", "eop", "eodds"]
demographic_predictors = ["DNN", "KNN"]

results_path = "./outputs/{}/results/averaged/".format(arg.dataset)
pareto_path = "./analysis/results/{}/pareto/".format(arg.dataset)
# base_dir = "./outputs/{}/results/".format(arg.dataset)
base_dir = "./outputs/{}/results/".format(arg.dataset)

if arg.model_name != "all":
    base_models = [arg.model_name]
    base_dir = "./outputs/{}/{}/results/".format(arg.dataset, arg.model_name)
    pareto_path = "./analysis/results/{}/pareto/{}/".format(
        arg.dataset, arg.model_name)

os.makedirs(results_path, exist_ok=True)
os.makedirs(pareto_path, exist_ok=True)
os.makedirs(base_dir, exist_ok=True)


def average_results(file_name, seeds):
    if len(seeds) == 0:
        return
    to_ret = {"acc": [], "unfair": [], "std_unfair": [], "std_acc": []}
    for file in seeds:
        df = pd.read_csv(file)
        df.sort_values('epsilon')
        to_ret["acc"].append(df.acc.values)
        to_ret["unfair"].append(df.unfair.values)
        eps = df["epsilon"].values

    accuracies = np.mean(to_ret["acc"], axis=0)
    unfairness = np.mean(to_ret["unfair"], axis=0)
    std_unfair = np.std(to_ret["unfair"], axis=0)
    std_acc = np.std(to_ret["acc"], axis=0)

    print(accuracies.shape, unfairness.shape)
    to_ret["acc"] = accuracies
    to_ret["unfair"] = unfairness
    to_ret["std_unfair"] = std_unfair
    to_ret["std_acc"] = std_acc
    to_ret["epsilon"] = eps
    # print(to_ret)
    df_to_save = pd.DataFrame(to_ret)

    out_file = "{}{}.csv".format(results_path, file_name)
    output_file = "{}{}.csv".format(pareto_path, file_name)
    df_to_save.to_csv(out_file, index=False)

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
    print(base_dir)
    if is_adv_debaising:
        for fair_metric in fair_metrics:
            if sentive_attr_type in ["predicted", "ours", "ours_w"]:
                for predictor in demographic_predictors:
                    file_name = "avd_debaising_model_{}_{}_{}".format(
                        fair_metric, sentive_attr_type, predictor)
                    # if os.path.isfile(file_path):
                    file_names.append(file_name)
            else:
                file_name = "avd_debaising_model_{}_{}".format(
                    fair_metric, sentive_attr_type)
                file_names.append(file_name)
    else:
        for model in base_models:
            for fair_metric in fair_metrics:
                if sentive_attr_type in ["predicted", "ours", "ours_w"]:
                    for predictor in demographic_predictors:
                        file_name = "{}_model_{}_{}_{}".format(
                            model, fair_metric, sentive_attr_type, predictor)
                        file_names.append(file_name)
                else:
                    file_name = "{}_model_{}_{}".format(
                        model, fair_metric, sentive_attr_type)
                    # if os.path.isfile(file_path):
                    file_names.append(file_name)

    for file_name in file_names:
        results[file_name] = []
        for r in range(10):
            file_path = "{}{}_{}.csv".format(base_dir, file_name, r)
            if os.path.isfile(file_path):
                results[file_name].append(file_path)
    return results


files = {}
for sensitive_ft_type in sensitive_ft_types:
    base_dir = "./outputs/{}/results/{}/".format(
        arg.dataset, sensitive_ft_type)
    clean = (get_result_sensitive_attr(base_dir=base_dir,
                                       is_adv_debaising=False, sentive_attr_type=sensitive_ft_type))
    # print(clean)
    files.update(clean)
    files.update(get_result_sensitive_attr(base_dir=base_dir,
                                           is_adv_debaising=True, sentive_attr_type=sensitive_ft_type))

for file in files:
    average_results(file, files[file])
