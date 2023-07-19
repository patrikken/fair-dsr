import itertools
import os
import numpy as np
import errno
import torch
from torch.utils.data import Sampler, BatchSampler, DistributedSampler
from tensorboardX import SummaryWriter
import pandas as pd

'''
    TensorBoard Data will be stored in './runs' path
'''
 
def train_test_split2(X, y, S, test_size=0.3):
    split_size = int(X.shape[0] * test_size)
    X_test, y_test, s_test = X[0:split_size,
                               :], y[0:split_size], S[0:split_size]
    X_train, y_train, s_train = X[split_size+1:,
                                  :], y[split_size+1:], S[split_size+1:]
    print(split_size)
    print(X_train.shape, y_train.shape)
    return torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test), torch.from_numpy(s_train), torch.from_numpy(s_test)


class DatasetLoader(torch.utils.data.Dataset):
    """ Create traning data iterator """

    def __init__(self, feature_X, label_y, sentive_a, transform=None):
        self.X = feature_X.float()
        self.y = label_y.unsqueeze(1).float()
        self.transform = transform
        self.A = None
        if sentive_a is not None:
            # if the sensitive attribute is not missing
            self.A = sentive_a.unsqueeze(1).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.X[idx, :].squeeze()
        if self.transform:
            X = self.transform(X)

        if self.A is not None:
            return X, self.y[idx, :].squeeze(), self.A[idx, :].squeeze()

        return X, self.y[idx, :]


class CustomSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, drop_last=False):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.batch_size = batch_size - secondary_batch_size
        self.secondary_batch_size = secondary_batch_size
        # self.drop_last = drop_last

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondar_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondar_batc
            for (primary_batch, secondar_batc) in zip(grouper(primary_iter, self.batch_size), grouper(secondar_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class DistributedBatchSampler(BatchSampler):
    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            dist_sampler = DistributedSampler(batch, **self.kwargs)
            dist_sampler_lst = list(dist_sampler)
            batch_out = [batch[x] for x in dist_sampler_lst]
            yield batch_out

    def __len__(self):
        return len(self.batch_sampler)


def get_adult_data_loader(include_y_in_x=True, batch_size=64):
    csv_path1 = "./preprocessing/adult_data1.csv"
    csv_path2 = "./preprocessing/adult_data2.csv"
    df_1 = pd.read_csv(csv_path1)
    del df_1['gender_ Male']
    df_2 = pd.read_csv(csv_path2)
     

    df = pd.concat([df_2, df_1], ignore_index=True)

     
    S = df['gender_ Male'].values
    print(df['gender_ Male'])
    del df['gender_ Male']
    if not include_y_in_x:
        X = df.drop('outcome_ >50K', axis=1).values
    else:
        X = df.values
     
    y = df['outcome_ >50K'].values

    X, y, S = torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(S)

    data = DatasetLoader(
        X, y.long(), S, transform=None)

    print(len(df_2), len(df_1), len(df))
    return data, len(df_2), len(df)

 