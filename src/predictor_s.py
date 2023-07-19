from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import entropy
import numpy as np
from torchmetrics import Accuracy

import copy
 


def entropy2(p):
    probs = torch.nn.functional.softmax(p, dim=0)
    return entropy(probs)


class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


class GaussianWarmpUp:
    def __init__(self, total_epoch):
        self.total_epoch = total_epoch

    def __call__(self, current_epoch):
        t_max = self.total_epoch
        # print(">>>", self.current_epoch)
        w = 0.29 * torch.exp(torch.tensor(1.23 * (1 - (current_epoch/t_max))))
        return w


class TeacherModel(nn.Module):
    def __init__(self, model, ema_param, mc_iteration=8, H=.3) -> None:
        super(TeacherModel, self).__init__()
        self.model = copy.deepcopy(model)
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iteration
        self.H = H
        self.ema_param = ema_param

    def ema_update(self, new_params) -> None:
        # update teacher's paramaters with EMA
        for current_params, ema_params in zip(new_params, self.model.parameters()):
            # print(param1.shape, param2.shape)
            ema_params.data = self.ema_param * current_params + \
                (1 - self.ema_param) * ema_params
        pass

    def entropy_estimation(self, batch):
        def enable_dropout(model):
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

        self.model.eval()

        # self.dropout.train()
        enable_dropout(self.model)
        x, _, _ = batch
        pred = [self.model(x).unsqueeze(0)
                for _ in range(self.mc_iteration)]
        mc_outs = torch.vstack(pred)
        entpy = []
        count_uncertain = 0
        batch_size = x.shape[0]

        highly_certain_idx = []
        for i in range(batch_size):
            # for each sample, compute its entropy based on MC dropout u_c = 1/T* sum(pt_c) pt_c is the probability at time t
            mc_out = mc_outs[:, i]

            p_class1 = mc_out.sum() / self.mc_iteration
            p_class2 = (1 - mc_out).sum() / self.mc_iteration

            # compute the entropy h_i = - sum(u_c * log(u_c))
            probs = torch.tensor([p_class1, p_class2])
            e = entropy2(probs)
            if e < self.H:
                count_uncertain += 1
                highly_certain_idx.append(i)
            entpy.append(e)

        # return the average uncertainty and the percentange highly confident samples thresholded by H
        # return torch.mean(torch.tensor(entpy)), highly_certain_idx
        return entpy, highly_certain_idx

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0):
        self.dropout.train()
        x, _, s = batch
        pred = [self.dropout(self.model(x)) for _ in range(self.mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        return pred

    def forward(self, x):
        x = self.model(x)
        return x.squeeze()


class DemographicPredictor(LightningModule):
    """ 
        params:
            ema_param: float. Exponential Moving Average (EMA) param for updating teachers weights. Default = 0.3
            input_size: int. Input size. Default = 97
            output_size: int. Output size. Default = 1
            data1: DataLoader. DataLoader of the dataset without sensitive informations
    """

    def __init__(self, input_size=97, output_size=1, lr=.001, ema_param=.99, betas=None, labelled_bs=64, consistency_w=.3, treshold_uncert=.3, total_epoch=50) -> None:
        super(DemographicPredictor, self).__init__()

        self.student = nn.Sequential(
            nn.Linear(input_size, 40),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(40, 25),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(25, 15),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(15, output_size)
        )

        self.treshold_uncert = treshold_uncert

        self.teacher = TeacherModel(
            self.student, ema_param, H=self.treshold_uncert)

        self.lr = lr
        self.consistency_w = 0
        self.use_consitency = consistency_w
        self.loss = nn.BCEWithLogitsLoss()
        self.consitency_loss = nn.MSELoss()

        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.teacher_acc = Accuracy(task="binary")
        self.labelled_bs = labelled_bs
        self.input_size = input_size
        self.total_epoch = total_epoch

        self.uncert_scheduler = GaussianWarmpUp(total_epoch=total_epoch)

    def set_uncertainty_treshold(self, H):
        self.treshold_uncert = H
        self.teacher.H = H

    def forward(self, x):
        x = self.student(x)
        return x.squeeze()

    def predict(self, x):
        # output = self.teacher(x)
        output = self(x)
        ans = torch.round(torch.sigmoid(output))
        return ans

    def training_step(self, batch, _):
        x, _, s = batch

        # x, s = x.squeeze(), s.squeeze()
        if len(x) < self.labelled_bs:
            return 0

        x_labelled, x_unlabelled = x[:self.labelled_bs,
                                     :], x[self.labelled_bs:, :]

        s_labelled, s_unlabelled = s[:self.labelled_bs], s[self.labelled_bs:]

        # print(s_labelled, s_unlabelled)

        output = self(x_labelled)

        loss = self.loss(output, s_labelled)

        # print("labeleled>", s_labelled)
        # print("unlabeleled>", s_unlabelled)

        self.train_acc.update(output, s_labelled.long())

        self.log("acc/train", self.train_acc,
                 prog_bar=True, on_epoch=True)
        self.log("loss/train", loss, on_epoch=True)

        # Evaluate teacher model
        with torch.no_grad():
            # self.teacher.eval()
            uncertainties, idx = self.teacher.entropy_estimation(
                (x, None, s_labelled))
            # self.teacher_acc.update(output.squeeze(), s.long())
            self.log("uncert_unlabeled", np.mean(uncertainties),
                     prog_bar=False, on_epoch=True, on_step=True)
            if len(idx) > 0:
                """ self.log("uncert_unlabeled_reliable_count", len(idx),
                         prog_bar=False, on_epoch=True, on_step=True) """

                teacher_out = self.teacher(x[idx, :])

        consistency_loss = 0

        if len(idx) > 0:
            student_out = self(x[idx, :])

            consistency_loss = self.consitency_loss(teacher_out, student_out)

            self.log("loss/consistency", consistency_loss,
                     prog_bar=False, on_epoch=True, on_step=True)

        if not self.use_consitency:
            return loss
        return loss + self.consistency_w * consistency_loss

    def update_consistency_weight(self):
        if not self.use_consitency:
            return 0
        t = self.current_epoch
        t_max = self.total_epoch
        # print(">>>", self.current_epoch)
        w = 0.1 * torch.exp(torch.tensor(-5 * (1 - (t/t_max))))
        self.log("loss/consistency_w", w,
                 prog_bar=False, on_epoch=True, on_step=False)

        self.consistency_w = w
        return w

    def update_treshold(self):
        t = self.current_epoch
        # print(">>>", self.current_epoch)
        u_max = torch.log10(torch.tensor(2))
        trhl = (3/4)*u_max * torch.exp(torch.tensor(-5 * (1 - (t/u_max))))
        self.log("loss/u_treshold", trhl,
                 prog_bar=False, on_epoch=True, on_step=False)

        # self.treshold_uncert = trhl
        return trhl

    def on_train_epoch_start(self):
        self.update_consistency_weight()
        self.treshold_uncert = self.uncert_scheduler(self.current_epoch)
        self.log("loss/u_treshold", self.treshold_uncert,
                 prog_bar=False, on_epoch=True, on_step=False)

    def validation_step(self, batch, _):
        x, _, s = batch

        # print(s)
        output = self(x)

        loss = self.loss(output, s)

        self.val_acc.update(output, s.long())

        self.log("acc/val", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("loss/val", loss)

        if not self.use_consitency:
            return loss
        # Evaluate teacher model
        with torch.no_grad():
            # self.teacher.eval()
            uncertainties, idx = self.teacher.entropy_estimation(batch)
            # self.teacher_acc.update(output.squeeze(), s.long())
            self.log("uncert_val", np.mean(uncertainties),
                     prog_bar=False, on_epoch=True)
            self.log("uncert_val_reliable_count", len(idx),
                     prog_bar=False, on_epoch=True)
            # self.log("mc_loss", mc_loss, prog_bar=False, on_epoch=True)

    def test_step(self, batch, _):
        x, y, s = batch

        output = self(x)

        loss = self.loss(output, s)

        self.test_acc.update(output, s.long())

        self.log("acc/test", self.test_acc, on_epoch=True)
        # self.log("test/loss", loss)

    def training_step_end(self, step_output):
        self.teacher.ema_update(self.parameters())
        return

    def configure_optimizers(self):
        optim = Adam(self.student.parameters(), self.lr)
        return optim
