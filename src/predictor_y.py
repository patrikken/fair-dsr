from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam, RMSprop
import torch
from torchmetrics import Accuracy
 
import torch.nn.functional as F
from metrics import DemParityMetric, EqualizedOdds, EqualOpportunity
from DRO.robust_loss import RobustLoss




class Classifier(LightningModule):
    """
        Gender predictor for the UCI Adult datatset

        params:
            use_robust: boolean. Use DRO loss if true.
            robust_method: enum. values `chi-square` or `cvar`
            input_size: int. Input size. Default = 97
            output_size: int. Output size. Default = 1
    """

    def __init__(self, input_size=97, output_size=1, lr=.001, betas=None, use_robust=False, robust_method='chi-square') -> None:
        super(Classifier, self).__init__()

        self.model = nn.Linear(input_size, output_size)   

        self.lr = lr
        self.robust_loss = RobustLoss(geometry=robust_method, size=1 if robust_method=='chi-square' else .9, reg=0.01)
        self.loss = nn.BCEWithLogitsLoss() 

        # metrics to logged 
        self.train_acc = Accuracy(task="binary", multiclass=False)
        self.val_acc = Accuracy(task="binary", multiclass=False)
        self.test_acc = Accuracy(task="binary", multiclass=False) 
        self.dp = DemParityMetric()

        self.dp_test = DemParityMetric()
        self.eo_test = EqualOpportunity()
        self.eod_test = EqualizedOdds()


        self.dp_train = DemParityMetric()

        self.use_robust = use_robust
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x).squeeze()
        sigmoid_output = self.sigmoid(x)
        return x, sigmoid_output

    def loss_fn(self, outputs, targets):
        if not self.use_robust:
            loss = self.loss(outputs, targets)
        else: 
            loss =  F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
            self.log("acc/n_train", loss.mean())
            loss = self.robust_loss(loss)
            self.log("acc/r_loss", loss)
        return loss


    def training_step(self, batch, _): 
        x, y, s = batch
         
        output, sigmoid_output = self(x)

        loss = self.loss_fn(output, y)

        self.train_acc.update(sigmoid_output, y.long())

        self.log("acc/train", self.train_acc,
                 prog_bar=True, on_epoch=True, on_step=False)
        self.log("loss/train", loss)

        self.dp_train.update(sigmoid_output, s)
        self.log("dp/train", self.dp_train, prog_bar=False,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, _):
        
        x, y, s = batch

        # print(s)
        output, sigmoid_output = self(x)

        loss = self.loss_fn(output, y)

        self.val_acc.update(sigmoid_output, y.long())

        self.dp.update(sigmoid_output, s)

        self.log("acc/val", self.val_acc, prog_bar=True)
        self.log("loss/val", loss, prog_bar=False)
        self.log("dp/val", self.dp, prog_bar=False)

    def test_step(self, batch, _):
        x, y, s = batch

        _, sigmoid_output = self(x)

        #loss = self.loss(output, y)

        self.test_acc.update(sigmoid_output, y.long())
        self.dp_test.update(sigmoid_output, s)
        self.eo_test.update(sigmoid_output, y, s)
        self.eod_test.update(sigmoid_output, y, s)

        self.log("acc/test", self.test_acc)
        self.log("dp/test", self.dp_test)
        self.log("eo/test", self.eo_test)
        self.log("eod/test", self.eod_test)
        # self.log("test/loss", loss)

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), self.lr)
        return optim
    
    def predict(self, x): 
        _, preds = self(x) 
        ans = (preds > .5).long()
        return ans
