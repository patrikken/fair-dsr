from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import entropy
import numpy as np
from torchmetrics import Accuracy
import torch
import copy


class FairDA(LightningModule):
    """
    params:
        input_size: int. Input size. Default = 97
        output_size: int. Output size. Default = 1
        lr: learning rate
        labelled_bs: sized of samples with sensitive attributes in the batch
    """

    def __init__(self, input_size=97, output_size=1, lr=0.001, labelled_bs=64) -> None:
        super(FairDA, self).__init__()
        self.latent_size = 40

        self.h1 = nn.Linear(input_size, self.latent_size)

        self.h2 = nn.Linear(input_size, self.latent_size)

        self.label_model = nn.Linear(self.latent_size, output_size)

        self.domain_model = nn.Linear(self.latent_size, output_size)

        self.sensitive_attrib_model = nn.Linear(self.latent_size, output_size)

        self.bias_model = nn.Linear(self.latent_size, output_size)

        self.automatic_optimization = False

        self.lr = lr

        self.loss = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.labelled_bs = labelled_bs
        self.input_size = input_size

        self.sigmoid = nn.Sigmoid()

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.alpha = 0.1
        self.beta = 0.1

    def forward(self, x):
        x = self.label_model(self.h2(x))
        return x.squeeze()

    def predict(self, x, threshold=0.5):
        output = self(x)
        # output = self(x)
        sigmoid = self.sigmoid(output)
        ans = (sigmoid > threshold).long()
        return ans

    def predict_prob(self, x):
        digits = self(x)
        return self.sigmoid(digits)

    def source_domain_step(self, x_source, s_source, x_labelled):
        latent = self.h1(x_source)

        output_s = self.sensitive_attrib_model(latent)

        loss_s = self.bce_loss(output_s.squeeze(), s_source)

        domain_labels = torch.cat(
            (torch.zeros(x_source.shape[0]), torch.ones(x_labelled.shape[0]))
        )
        latent_target = self.h1(x_labelled)

        in_domain = torch.cat((latent, latent_target), 0)

        output_domain = self.domain_model(in_domain)
        loss_domain = self.bce_loss(output_domain.squeeze(), domain_labels)

        self.log("loss/loss_A", loss_s, on_epoch=True)
        return loss_s - self.alpha * loss_domain

    def target_domain_step(self, x_target, y_target):
        latent = self.h2(x_target)

        out_label = self.label_model(latent)

        with torch.no_grad():
            s_pred = self.sensitive_attrib_model(self.h1(x_target))

        bias_out = self.bias_model(latent)

        label_loss = self.bce_loss(out_label.squeeze(), y_target)

        bias_loss = self.bce_loss(bias_out, s_pred)

        self.train_acc.update(out_label.squeeze(), y_target.long())

        self.log("acc/train", self.train_acc, prog_bar=True, on_epoch=True)
        self.log("loss/train", label_loss, on_epoch=True)
        self.log("loss/bias", bias_loss, on_epoch=True)

        return label_loss - self.beta * bias_loss

    def training_step(self, batch, _):
        (
            optim_label,
            optim_sensitive_attrib,
            optim_bias,
            optim_domain,
        ) = self.optimizers()

        optim_sensitive_attrib.zero_grad()
        optim_domain.zero_grad()
        optim_label.zero_grad()
        optim_bias.zero_grad()

        x, y, s = batch

        # x, s = x.squeeze(), s.squeeze()
        if len(x) < self.labelled_bs:
            return 0

        x_source, x_target = x[: self.labelled_bs, :], x[self.labelled_bs :, :]

        s_source, _ = s[: self.labelled_bs], s[self.labelled_bs :]

        _, y_target = y[: self.labelled_bs], y[self.labelled_bs :]

        ### TRAIN SOURCE DOMAINE

        L1 = self.source_domain_step(x_source, s_source, x_target)
        L2 = self.target_domain_step(x_target, y_target)

        loss = L1 + L2

        self.manual_backward(loss)

        optim_sensitive_attrib.step()
        optim_domain.step()
        optim_label.step()
        optim_bias.step()
        return

    def validation_step(self, batch, _):
        x, y, _ = batch

        # print(s)
        output = self(x)

        loss = self.loss(output, y)

        self.val_acc.update(output, y.long())

        self.log("acc/val", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("loss/val", loss)

    def test_step(self, batch, _):
        x, y, _ = batch

        output = self(x)

        loss = self.loss(output, y)

        self.test_acc.update(output, y.long())

        self.log("acc/test", self.test_acc, on_epoch=True)
        # self.log("test/loss", loss)

    def configure_optimizers(self):
        optim_label = Adam(self.label_model.parameters(), self.lr)
        optim_bias = Adam(self.bias_model.parameters(), self.lr)
        optim_domain = Adam(self.domain_model.parameters(), self.lr)
        optim_sensitive_attrib = Adam(self.sensitive_attrib_model.parameters(), self.lr)
        return optim_label, optim_sensitive_attrib, optim_bias, optim_domain
