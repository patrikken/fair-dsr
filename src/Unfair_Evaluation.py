from argparse import ArgumentParser
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score
import pandas as pd
from fairlearn.reductions import DemographicParity, FalsePositiveRateParity, TruePositiveRateParity, EqualizedOdds
from sklearn.model_selection import train_test_split
import os

import torch
from torch import nn
import numpy as np
from datasets import get_datasets_tp

from metrics import equal_opportunity
from helper import get_base_models

parser = ArgumentParser()

parser.add_argument("--base_model", type=str, default='lr',
                    help="Base classifier for reduction methods, lr:LosgisticRegression, rf:RandomForest")
parser.add_argument("--seed", type=int, default=1,
                    help="Number of seeds")
parser.add_argument("--dataset", type=str, default='new_adult',
                    help="3 datasets are available: adult, new_adult, acs_employment")

arg = parser.parse_args()


# seed

np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

fair_method_map = {
    'dp': DemographicParity,
    'fpr': FalsePositiveRateParity,
    'eop': TruePositiveRateParity,
    'eodds': EqualizedOdds
}

fair_metrics_map = {
    'dp': demographic_parity_difference,
    # 'fpr' : false_negative_rate,
    'eop': equal_opportunity,
    'eodds': equalized_odds_difference
}

fair_metrics_name_map = {
    'dp': 'demographic_parity',
    'eodds': 'equalized_odds'
}

basemodel_keys = ['lr', 'gbm', 'rf', 'avd_debaising']


basemodel_map = get_base_models(arg.dataset)



def train_test_split2(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size,
                               :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size+1:,
                                  :], y[split_size+1:], S[split_size+1:]
    print(split_size)
    print(X_train.shape, y_train.shape)
    return X_train, X_test, y_train, y_test, s_train, s_test


class NN(nn.Module):
    def __init__(self, input_size=97, out_size=1) -> None:
        super(NN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, out_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.network(x)
        return x

datasets = get_datasets_tp()

data1, _ = datasets[arg.dataset]()

X, y, s = data1

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.2, random_state=arg.seed)

print(X.shape, X_train.shape, X_test.shape)


def fit(base_model, fairness):

    clf = basemodel_map[base_model]

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    to_ret = {}
    for fairness in fair_metrics_map.keys():
        unfairness_measure = fair_metrics_map[fairness]

        unfair = unfairness_measure(
            y_true=y_test, y_pred=y_pred, sensitive_features=s_test)
        to_ret[fairness] = [unfair]
    accur = accuracy_score(y_true=y_test, y_pred=y_pred)
    to_ret.update({'acc': [accur]})
    return to_ret


def main(args):

    out_path = 'analysis/results/{}/averaged/'.format(args.dataset)

    os.makedirs(out_path, exist_ok=True)

    for base_model in basemodel_keys:
        print("---------------------------->>> base model = {}".format(base_model))
        out_file = '{}/unfair_{}_model.csv'.format(out_path, base_model)
        res = fit(base_model, args.base_model)
        print(res)
        df = pd.DataFrame(res)
        df.to_csv(out_file, encoding='utf-8', index=False)


if __name__ == "__main__":

    main(arg)
