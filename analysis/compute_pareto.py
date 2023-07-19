import csv
import numpy as np
import pandas as pd
from six.moves import xrange
import os
import sys

import argparse
import csv

from ndf import is_pareto_efficient


# parser initialization
parser = argparse.ArgumentParser(description='Analysis of the results')
parser.add_argument('--dataset', type=str, default='adult',
                    help='adult, compas, default_credit, marketing')
parser.add_argument('--set', type=str, default='test',
                    help='Set to use, train or test')
parser.add_argument('--metric', type=str, default='dp', help='dp, eod, eop')


explainers_map = {
    'rl': '_rl',
    'lm': '_lm',
    'dt': '_dt'
}

metrics_map = {
    1: 'statistical_parity',
    3: 'predictive_equality',
    4: 'equal_opportunity',
    5: 'equalized_odds',
}

eval_group_map = {
    'sg': "Members",
    'test': "Non-Members"
}

metric_to_use = {
    'train': ('acc/train', 'dp/train'),
    'test': ('acc/test', 'dp/test'),
    'val': ('acc/val', 'dp/val'),
    'epsilon': 'epsilon'
}


def compute_front(df_average, output_file, set):
    accuracy, fairness = metric_to_use[set]
    accuracies = df_average[accuracy]
    fairness = df_average[fairness] 

    eps = df_average[metric_to_use['epsilon']]

    errors = 1.0 - accuracies
    pareto_input = [[error, unfair]
                    for (error, unfair) in zip(errors, fairness)]
    pareto_input = np.array(pareto_input)
    msk = is_pareto_efficient(pareto_input)

    df = pd.DataFrame()
    df['fairness'] = [fairness[i] for i in xrange(len(fairness)) if msk[i]]
    df['accuracy'] = [1.0 - errors[i]
                      for i in xrange(len(accuracies)) if msk[i]]
    df['epsilon'] = [eps[i] for i in xrange(len(eps)) if msk[i]] 
    df.to_csv(output_file, encoding='utf-8', index=False)


def main():
    args = parser.parse_args()
    dataset = args.dataset
    set_type = args.set 

    models = ["clean", "predicted", "ours"] 
    save_dir = "./analysis/results/{}/pareto_fair_batch/".format(dataset)

    os.makedirs(save_dir, exist_ok=True)
    for model in models:
        input_file = "./outputs/{}/runs/model_{}_{}.csv".format(
            dataset, model, args.metric)
        df_average = pd.read_csv(input_file)

        output_file_train = '{}/model_{}_{}_{}.csv'.format(
            save_dir, model, set_type, args.metric)
        compute_front(df_average, output_file_train, set_type)

        output_file_test = '{}/model_{}_{}_{}.csv'.format(
            save_dir, model, set_type, args.metric)
        compute_front(df_average, output_file_test, set_type)


if __name__ == '__main__':
    sys.exit(main())
