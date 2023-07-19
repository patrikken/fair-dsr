import random
from pytorch_lightning import LightningDataModule
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from utils import CustomSampler, DistributedBatchSampler

from torch.utils.data.sampler import BatchSampler

from utils import DatasetLoader, train_test_split2

import numpy as np
from sklearn.metrics import accuracy_score

 
from datasets import get_old_adult, get_adult


seed = 42


def worker_init_fn(worker_id):
    random.seed(seed+worker_id)


class BaseDataModule(LightningDataModule):
    """
    Data model for the UCI Adult datatset

    Params:
        include_y_in_x: bool. When true adds the target to the input feature space
    """

    def __init__(self, csv_path, n_features, batch_size, num_workers=4, include_y_in_x=True, model=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.include_y_in_x = include_y_in_x

        self.n_feature = n_features

        self.sampler = None

        self.model = model

        if self.include_y_in_x is True:
            self.n_feature += 1

        self.seed = 42

    def prepare_data(self):
        return

    def get_data(self):
        return None, None, None

    def setup(self, stage):
        X, y, S = self.get_data()
        X_train, X_test, y_train, y_test, S_train, S_test = train_test_split2(
            X, y, S, test_size=0.3)

        train_dataset = DatasetLoader(
            X_train, y_train.long(), S_train.long(), transform=None)

        # use 20% of training data for validation
        train_set_size = int(len(train_dataset) * 0.8)
        valid_set_size = len(train_dataset) - train_set_size

        # split the train set into two
        self.train_data, self.val_data = random_split(
            train_dataset, [train_set_size, valid_set_size], generator=self.seed)

        self.test_data = DatasetLoader(
            X_test, y_test.long(), S_test.long(), transform=None)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self):

        if self.sampler is None:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        return DataLoader(self.train_data, num_workers=self.num_workers, batch_sampler=self.sampler, pin_memory=True, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
 
class DataModelMissingSensitiveAtt(BaseDataModule):
    """
    Data model for the UCI Adult datatset joint dataset for with missing sensitiveAttribute

    Params:
        include_y_in_x: bool. When true adds the target to the input feature space
    """

    def __init__(self, data1, data2, data1_test, val_size=0, batch_size=128, num_workers=4, include_y_in_x=True, model=None, labeled_bs=64):
        super().__init__(csv_path="", batch_size=batch_size,
                         num_workers=num_workers, include_y_in_x=include_y_in_x, model=model, n_features=96)

        assert batch_size > labeled_bs

        self.labeled_bs = labeled_bs
        self.data1 = data1
        self.data2 = data2
        self.data1_test = data1_test
        self.val_size = val_size
        self.n_feature = data1[0].shape[1]
        if include_y_in_x is True:
            self.n_feature += 1

    def setup(self, stage):

        def converter_to_tensor(data):
            X, y, S = data
            return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(S)

        X_d1, y_d1, S_d1 = converter_to_tensor(self.data1)
        X_d2, y_d2, S_d2 = converter_to_tensor(self.data2)
        X_test, y_test, S_test = converter_to_tensor(self.data1_test)
        S_d1[True] = torch.tensor(-1.0)
        X = torch.vstack((X_d2, X_d1))
        print(">>>", X.shape, X_d2.shape, X_d1.shape)
        y = torch.vstack((y_d2.unsqueeze(dim=1), y_d1.unsqueeze(dim=1)))
        s = torch.vstack((S_d2.unsqueeze(dim=1), S_d1.unsqueeze(dim=1)))

        if self.include_y_in_x is True:
            X = torch.hstack((X, y))
            y_test = y_test.unsqueeze(dim=1)
            X_test = torch.hstack((X_test, y_test))
        self.train_data = DatasetLoader(
            X, y, s, transform=None)

        val_dataset = DatasetLoader(
            X_test, y_test, S_test, transform=None)

        # use 20% of test data for validation
        val_set_size = int(len(val_dataset) * self.val_size)
        test_set_size = len(val_dataset) - val_set_size

        # split the train set into two
        self.val_data, self.test_data = random_split(
            val_dataset, [val_set_size, test_set_size])

        labeled_range, total_range = len(X_d2), len(X)
        # print(">>>*", labeled_range, total_range)
        labeled_idxs = list(range(labeled_range))
        unlabeled_idxs = list(range(labeled_range, total_range))

        sampler = CustomSampler(
            primary_indices=labeled_idxs, secondary_indices=unlabeled_idxs, batch_size=self.batch_size, secondary_batch_size=self.batch_size - self.labeled_bs, drop_last=True)

        # batch_sampler = DistributedBatchSampler(sampler)
        self.sampler = sampler
        return

    def val_dataloader(self):
        if self.val_size == 0:
            return None
        return DataLoader(self.val_data, batch_size=self.labeled_bs, shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self):

        if self.sampler is None:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        return DataLoader(self.train_data, num_workers=self.num_workers, batch_sampler=self.sampler, pin_memory=True, worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.labeled_bs, shuffle=False, num_workers=self.num_workers)
 