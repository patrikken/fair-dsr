from argparse import ArgumentParser
from fairlearn.reductions import DemographicParity
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score
import pandas as pd
from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity,
    TruePositiveRateParity,
    EqualizedOdds,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import numpy as np
from datasets import get_datasets_tp

from predictor_s import DemographicPredictor
from knn_imputer import knn_impute
from metrics import equal_opportunity
import mpi4py.rc
from mpi4py import MPI
import os
from helper import get_base_models
from mapie.classification import MapieClassifier
from sklearn.calibration import CalibratedClassifierCV
import random

parser = ArgumentParser()

parser.add_argument(
    "--fair_metric",
    type=str,
    default="dp",
    choices=["dp", "eod", "eodds"],
    help="Fairness metric. `dp`, `eod`, `eodds`",
)

parser.add_argument(
    "--base_model",
    type=str,
    default="lr",
    choices=["lr", "rf", "gbm"],
    help="Base classifier for reduction methods, lr:LosgisticRegression, rf:RandomForest, gbm:GradientBoostingClassifier",
)

parser.add_argument(
    "--sensitive_feature_type",
    type=str,
    default="clean",
    choices=["clean", "ours", "predicted"],
    help="Use clean or predicted sensitive features. `clean`, `ours` or `predicted`",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="adult",
    choices=["compas_race", "lsac_sex", "celeba", "celeba_attract"],
    help="The dataset to be used",
)

parser.add_argument(
    "--demographic_predictor",
    type=str,
    default="DNN",
    choices=["DNN", "KNN"],
    help="Model used to infer the sensitive attribute from the related feature. `DNN`, `KNN`",
)

parser.add_argument(
    "--is_adv_method",
    action="store_true",
    help="Whether to use adversarial debiasing",
)

parser.add_argument(
    "--cp_alpha",
    type=float,
    default=0.05,
    help="The value of the parameter alpha controlling the coverage of the prediction set",
)

parser.add_argument("--seed", type=int, default=1, help="Seed number")

arg = parser.parse_args()


# seed

np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

fair_method_map = {
    "dp": DemographicParity,
    "eop": TruePositiveRateParity,
    "eodds": EqualizedOdds,
}

fair_metrics_map = {
    "dp": demographic_parity_difference,
    "fpr": false_negative_rate,
    "eop": equal_opportunity,
    "eodds": equalized_odds_difference,
}

fair_metrics_name_map = {"dp": "demographic_parity", "eodds": "equalized_odds"}


# epsilons
epsilon_range = np.arange(0.701, 0.991, 0.004)
base = [0.0, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + list(epsilon_range) + [0.9999]
epsilons = [round(x, 3) for x in epsilon_range]  # 300 values
# epsilons = [0.0, 1.0]  # 0.7, 0.801, 0.985, 0.989, 0.989,
# epsilons = [0.0, 1.0]  # 0.7, 0.801, 0.985, 0.989, 0.989,


basemodel_map = get_base_models

datasets = get_datasets_tp()


def get_predicted_sensitive_attribute(
    X, y, return_Reliable_S=False, predict_prob=False, get_cls=False
):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    input_size = X.shape[1]

    if arg.dataset == "new_adult":
        pretrained_path = "./pretrained/new_adult_consistency_loss2.ckpt"
        input_size = 52
        # add the target in the input to predict the sensitive attrbutes
        X = torch.hstack([X, y.float().unsqueeze(1)])
        treshold_uncert = 0.6
    elif arg.dataset == "compas":
        pretrained_path = "./pretrained/compas_gender.ckpt"
        # input_size = X.shape[1]
        treshold_uncert = 0.6
    elif arg.dataset == "compas_race":
        pretrained_path = "./pretrained/compas_race.ckpt"
        # input_size = X.shape[1]
        treshold_uncert = 0.6
    elif arg.dataset == "lsac":
        pretrained_path = "./pretrained/lsac_race_c.ckpt"
        treshold_uncert = 0.06  # 0.01106266
    elif arg.dataset == "lsac_sex":
        pretrained_path = "./pretrained/lsac_sex.ckpt"
        treshold_uncert = 0.66
    elif arg.dataset == "celeba":
        pretrained_path = "./pretrained/celebA.ckpt"
        treshold_uncert = 0.3
    elif arg.dataset == "celeba_attract":
        pretrained_path = "./pretrained/celeba_attract.ckpt"
        treshold_uncert = 0.3
    else:
        pretrained_path = "./pretrained/adult_pretrained_demographic.ckpt"
        input_size = 96
        treshold_uncert = 0.3

    demographic_predictor = DemographicPredictor.load_from_checkpoint(
        pretrained_path, treshold_uncert=treshold_uncert, input_size=input_size
    )

    with torch.no_grad():
        demographic_predictor.eval()
        s_pred = demographic_predictor.predict(X)
        if return_Reliable_S:
            entropies, idx = demographic_predictor.teacher.entropy_estimation(
                (X, None, None)
            )

            return s_pred.long().detach().numpy(), idx, np.array(entropies)
        if predict_prob:
            s_pred, s_prob = demographic_predictor.predict_prob(X)
            return s_prob, s_pred

        if get_cls:
            s_pred, s_prob = demographic_predictor.predict_prob(X)
            return s_prob, s_pred, demographic_predictor
    return s_pred.detach().numpy()


# load the dataset with sensitive attributes (data1) and without data2
data1, data2 = datasets[arg.dataset]()

X, y, s = data1

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.3, random_state=arg.seed
)
sample_weights = None

if arg.sensitive_feature_type == "cp":

    def get_certainty_set(y_ps_score):
        alpha_idx = 0
        uncertain_prediction = []
        certain_prediction = []
        for i in range(len(y_ps_score)):
            prediction_set_size = y_ps_score[i].sum(axis=0)[alpha_idx]
            if prediction_set_size == 1:
                certain_prediction.append(i)
            else:
                uncertain_prediction.append(i)
        return certain_prediction, uncertain_prediction

    _, X_val, _, y_val, _, s_val = train_test_split(
        X_train, y_train, s_train, test_size=0.1, random_state=arg.seed
    )
    val_idx = random.choices(range(len(y_train)), k=int(len(y_train) * 0.2))

    X_t, y_t, s_t = data2
    clf = LogisticRegression(max_iter=1400).fit(X_t, s_t)
    s_pred = clf.predict(X_train)
    s_pred_proba = clf.predict_proba(X_train)
    s_pred_proba_max = np.max(s_pred_proba, axis=1)

    calib = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
    X_c1, X_c2, s_c1, s_c2 = train_test_split(
        X_train[val_idx, :], s_train[val_idx], test_size=0.5
    )
    calib.fit(X_c1, s_c1)

    mapie_clf = MapieClassifier(
        estimator=calib, method="lac", cv="prefit", random_state=42
    )
    mapie_clf.fit(X_c2, s_c2)

    alpha = [arg.cp_alpha]
    y_pred_score, y_ps_score = mapie_clf.predict(X_train, alpha=alpha)

    certain_prediction_idx, _ = get_certainty_set(y_ps_score)
    s_train = y_pred_score[certain_prediction_idx]
    X_train = X_train[certain_prediction_idx, :]
    y_train = y_train[certain_prediction_idx]

elif arg.sensitive_feature_type == "ours":
    s_pred, idx, uncert = get_predicted_sensitive_attribute(
        X_train, y_train, return_Reliable_S=True
    )
    s_train = s_pred[idx]
    X_train = X_train[idx, :]
    y_train = y_train[idx]

elif arg.sensitive_feature_type == "predicted":
    if arg.demographic_predictor == "DNN":
        s_pred = get_predicted_sensitive_attribute(X_train, y_train)
    elif arg.demographic_predictor == "KNN":
        X_t, y_t, s_t = data2
        s_pred = knn_impute(X_t, s_t, X_train)
    s_train = s_pred


def fit(epsilon, fairness, base_model, is_adv_devaising=True):
    if is_adv_devaising:
        adv_input_size = 2 if fairness == "eodds" else 1
        mitigator = AdversarialFairnessClassifier(
            constraints=fair_metrics_name_map[fairness],
            backend="torch",
            predictor_model=[X_train.shape[1], "sigmoid"],
            adversary_model=[adv_input_size, "sigmoid"],
            epochs=50,
            alpha=epsilon,
        )
        # print(X_train[0])
    else:
        clf = basemodel_map(arg.dataset, arg.seed)[base_model]
        constraint = fair_method_map[fairness](difference_bound=1.0 - epsilon)
        mitigator = ExponentiatedGradient(clf, constraint)

    mitigator.fit(X_train, y_train, sensitive_features=s_train)

    y_pred = mitigator.predict(X_test)

    unfairness_measure = fair_metrics_map[fairness]

    unfair = unfairness_measure(y_true=y_test, y_pred=y_pred, sensitive_features=s_test)
    accur = accuracy_score(y_true=y_test, y_pred=y_pred)

    return unfair, accur


def process_result(epsilons, results):
    results_dict = {"unfair": [], "acc": [], "epsilon": epsilons}

    for res in results:
        unfair, acc = res
        results_dict["acc"].append(acc)
        results_dict["unfair"].append(unfair)

    return results_dict


def split(container, count):
    return [container[_i::count] for _i in range(count)]


def main(args):
    mpi4py.rc.threads = False

    is_adv_method = args.is_adv_method is True

    if is_adv_method and args.fair_metric not in fair_metrics_name_map.keys():
        print(
            "============= {} not supported for Adversarial debaising ".format(
                args.fair_metric
            )
        )
        return None

    COMM = MPI.COMM_WORLD

    print("COMM_SIZE = ", COMM.size)

    if COMM.rank == 0:
        jobs = split(epsilons, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)

    model_name = "avd_debaising" if is_adv_method else args.base_model
    out_path = "outputs/{}/results/{}".format(args.dataset, args.sensitive_feature_type)

    sensitive_feature_type = args.sensitive_feature_type

    if args.sensitive_feature_type == "cp":
        out_path += f"_{args.cp_alpha}"
        sensitive_feature_type += f"_{args.cp_alpha}"

    if args.use_uncertain:
        sensitive_feature_type += f"_uncertain"

    out_file = "{}/{}_model_{}_{}{}_{}.csv".format(
        out_path,
        model_name,
        args.fair_metric,
        sensitive_feature_type,
        (
            "_{}".format(args.demographic_predictor)
            if args.sensitive_feature_type != "clean"
            else ""
        ),
        arg.seed,
    )

    os.makedirs(out_path, exist_ok=True)
    results = []

    for epsilon in jobs:
        print("---------------------------->>> epsilon = {}".format(epsilon))

        res = fit(epsilon, args.fair_metric, args.base_model, is_adv_method)
        results.append(res)

    # Gather results on rank 0.
    results = MPI.COMM_WORLD.gather(results, root=0)
    if COMM.rank == 0:
        results = [_i for temp in results for _i in temp]
        processed = process_result(epsilons, results)
        df = pd.DataFrame(processed)
        df.to_csv(out_file, encoding="utf-8", index=False)
        print(out_file)


if __name__ == "__main__":
    main(arg)
