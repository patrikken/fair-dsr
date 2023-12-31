from argparse import ArgumentParser
from fairlearn.reductions import DemographicParity
from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_negative_rate
from sklearn.metrics import accuracy_score
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, FalsePositiveRateParity, TruePositiveRateParity, EqualizedOdds

import torch
from torch import nn
import numpy as np
from datasets import get_datasets_tp

from predictor_s import DemographicPredictor
from metrics import equal_opportunity
import mpi4py.rc
from mpi4py import MPI
from helper import get_base_models
import os

parser = ArgumentParser()

parser.add_argument('--fair_metric', type=str, default='dp',
                    help='Fairness metric. `dp`, `eod`, `eodds`, `fpr`')

parser.add_argument("--base_model", type=str, default='lr',
                    help="Base classifier for reduction methods, lr:LosgisticRegression, rf:RandomForest")

parser.add_argument("--treshold_uncert", type=float, default=.3,
                    help="Uncertainty threshold")

parser.add_argument("--sensitive_feature_type", type=str, default='ours',
                    help="Use clean or predicted sensitive features. `clean`, `ours` or `predicted`")

parser.add_argument("--dataset", type=str, default='adult',
                    help="Two datasets are available: Adult")

parser.add_argument("--demographic_predictor", type=str, default=None,
                    help="Model used to infer the sensitive attribute from the related feature. `DNN`, `KNN`")

parser.add_argument('--is_adv_method', action='store_true',
                    help='Use adversarial method if True. Else use Reduction methods')

parser.add_argument("--seed", type=int, default=1,
                    help="Number of seeds")

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
    'fpr': false_negative_rate,
    'eop': equal_opportunity,
    'eodds': equalized_odds_difference
}

fair_metrics_name_map = {
    'dp': 'demographic_parity',
    'eodds': 'equalized_odds'
}


# epsilons
epsilon_range = np.arange(0.701, 0.991, 0.004)
base = [0.0, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epsilon_range = base + list(epsilon_range) + [0.9999]
epsilons = [round(x, 3) for x in epsilon_range]  # 300 values
#epsilons = [0.0, 1.0]

basemodel_map = get_base_models(arg.dataset, arg.seed)

datasets = get_datasets_tp()


def train_test_split2(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size,
                               :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size+1:,
                                  :], y[split_size+1:], S[split_size+1:]
    print(split_size)
    print(X_train.shape, y_train.shape)
    return X_train, X_test, y_train, y_test, s_train, s_test


def get_predicted_sensitive_attribute(X, y, return_Reliable_S=False):
    print(">>>>>>", X.shape)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    input_size = X.shape[1]
    if arg.dataset == 'new_adult':
        pretrained_path = './pretrained/new_adult_consistency.ckpt'
        input_size = 51
        # add the target in the input to predict the sensitive attrbutes
        # X = torch.hstack([X, y.float().unsqueeze(1)]) 
    elif arg.dataset == "compas":
        pretrained_path = './pretrained/compas_race.ckpt' 
    elif arg.dataset == "lsac":
        pretrained_path = './pretrained/lsac_race.ckpt' 
    elif arg.dataset == "compas_race": 
        pretrained_path = './pretrained/compas_race.ckpt'  
    elif arg.dataset == "lsac_sex":
        pretrained_path = './pretrained/lsac_sex.ckpt' 
    else:
        pretrained_path = './pretrained/adult_pretrained_demographic.ckpt'
        input_size = 96

    demographic_predictor = DemographicPredictor.load_from_checkpoint(
        pretrained_path, treshold_uncert=arg.treshold_uncert, input_size=input_size)

    with torch.no_grad():
        demographic_predictor.eval()
        s_pred = demographic_predictor.predict(X)
        if return_Reliable_S:
            entropies, idx = demographic_predictor.teacher.entropy_estimation(
                (X, None, None))

            print("\n  Mean entropy", np.mean(entropies))
            print("\n  Low uncertainty proportion = {}/{}   {}%".format
                  (len(idx), X.shape[0], len(idx)/X.shape[0]))
            return s_pred, idx, np.array(entropies)
    return s_pred.detach().numpy()


def fit(epsilon, fairness, base_model, is_adv_devaising=True):

    if is_adv_devaising:
        adv_input_size = 2 if fairness == 'eodds' else 1
        mitigator = AdversarialFairnessClassifier(constraints=fair_metrics_name_map[fairness], predictor_model=NN(
            input_size=X_train.shape[1]), adversary_model=NN(input_size=adv_input_size), epochs=50, alpha=epsilon)
    else:
        clf = basemodel_map[base_model]
        # clf = clf.set_params(model__sample_weights = sample_weights)
        constraint = fair_method_map[fairness](difference_bound=1.0 - epsilon)
        mitigator = ExponentiatedGradient(clf, constraint)

    mitigator.fit(X_train, y_train, sensitive_features=s_train)

    y_pred = mitigator.predict(X_test)

    unfairness_measure = fair_metrics_map[fairness]

    unfair = unfairness_measure(
        y_true=y_test, y_pred=y_pred, sensitive_features=s_test)
    accur = accuracy_score(y_true=y_test, y_pred=y_pred)

    return unfair, accur


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


data1, data2 = datasets[arg.dataset]()

X, y, s = data1

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split2(X, y, s)
sample_weights = None

print("===================== Predicting and filtering the sensitive attribute")
s_pred, idx, _ = get_predicted_sensitive_attribute(
    X_train, y_train, return_Reliable_S=True)

print("///////////////////// Accuracy predicted target = ", accuracy_score(
    s_train, s_pred))
print(s_pred)
s_train = s_pred[idx]  # replace by the  prediced
X_train = X_train[idx, :]
y_train = y_train[idx]


def process_result(epsilons, results):
    results_dict = {
        'unfair': [],
        'acc': [],
        'epsilon': epsilons
    }

    for res in results:
        unfair, acc = res
        results_dict['acc'].append(acc)
        results_dict['unfair'].append(unfair)

    return results_dict


def split(container, count):
    return [container[_i::count] for _i in range(count)]


def main(args):
    mpi4py.rc.threads = False

    is_adv_method = args.is_adv_method is True

    if is_adv_method and args.fair_metric not in fair_metrics_name_map.keys():
        print("============= {} not supported for Adversarial debaising ".format(
            args.fair_metric))
        return None

    print(">>>>>>>>>"*20)

    COMM = MPI.COMM_WORLD

    print("COMM_SIZE = ", COMM.size)

    if COMM.rank == 0:
        jobs = split(epsilons, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)

    model_name = 'avd_debaising' if is_adv_method else args.base_model
    out_path = 'outputs/{}/ablation/{}_{}'.format(args.dataset,
                                                  args.sensitive_feature_type, args.treshold_uncert)
    out_file = '{}/{}_model_{}_{}{}_{}.csv'.format(
        out_path, model_name, args.fair_metric, args.sensitive_feature_type, "_DNN" if arg.demographic_predictor == "DNN" else "", args.seed)

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
        df.to_csv(out_file, encoding='utf-8', index=False)


if __name__ == "__main__":

    main(arg)
