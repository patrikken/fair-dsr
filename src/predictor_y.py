from argparse import ArgumentParser, Namespace
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import NeptuneLogger
import torch
from torch import nn, cat
from torch.optim import Adam
import torch.nn.functional as F

from torchmetrics import Accuracy

from data_module import AdultDataModel, GermanDataModel

from scipy.stats import entropy
from metrics import DemParityMetric

from predictor_s import DemographicPredictor


class Classifier(LightningModule):
    """
        Gender predictor for the UCI Adult datatset

        params:
            ema_param: float. Exponential Moving Average (EMA) param for updating teachers weights. Default = 0.3
            input_size: int. Input size. Default = 97
            output_size: int. Output size. Default = 1
    """

    def __init__(self, input_size=97, output_size=1, lr=.001, betas=None) -> None:
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 60),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(60, output_size) 
        )

        self.lr = lr

        self.loss = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.teacher_acc = Accuracy(task="binary")
        self.dp = DemParityMetric()
        self.dp_test = DemParityMetric()
        self.dp_train = DemParityMetric()

    def forward(self, x):
        x = self.model(x)
        return x.squeeze()

    def training_step(self, batch, _):
        x, y, s = batch
        # x, y, s = x.squeeze(), y.squeeze(), s.squeeze()
        # print(x.shape, s)
        output = self(x)

        #

        loss = self.loss(output, y)

        self.train_acc.update(output, y.long())

        self.log("acc/train", self.train_acc,
                 prog_bar=True, on_epoch=True, on_step=False)
        self.log("loss/train", loss)

        self.dp_train.update(output, s)
        self.log("dp/train", self.dp_train, prog_bar=False,
                 on_epoch=True, on_step=False)

        return loss

    """ def training_epoch_end(self, outputs) -> None:
        # update teacher's paramaters with EMA
        # for current_params, ema_params in zip(self.parameters(), self.teacher.parameters()):
        # print(param1.shape, param2.shape)
        #    ema_params.data = self.ema_param * current_params + \
        #        (1 - self.ema_param) * ema_params
        pass """

    def validation_step(self, batch, _):
        x, y, s = batch

        # print(s)
        output = self(x)

        loss = self.loss(output, y)

        self.val_acc.update(output, y.long())

        self.dp.update(output, s)

        self.log("acc/val", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("loss/val", loss, prog_bar=True, on_epoch=True)
        self.log("dp/val", self.dp, prog_bar=True, on_epoch=True)

    def test_step(self, batch, _):
        x, y, s = batch

        output = self(x)

        loss = self.loss(output, y)

        self.test_acc.update(output, y.long())
        self.dp_test.update(output, s)

        self.log("acc/test", self.test_acc, on_epoch=True)
        self.log("dp/test", self.dp_test, prog_bar=True, on_epoch=True)
        # self.log("test/loss", loss)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), self.lr)
        return optim
