"""
   Pytorch Lightning Implementation of Adversarially Reweighted Learning model by Lahoti et al. (2020. 
"""
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
import torch
from torchmetrics import Accuracy 

from metrics import DemParityMetric, EqualizedOdds, EqualOpportunity
import torch.nn.functional as F

class AdversaryNN(nn.Module):
    def __init__(self, input_size=97, output_size=1):
        super(AdversaryNN, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
         
        self.sigmoid = nn.Sigmoid()
         
    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(x)
        x_mean = torch.mean(x)

        x = x / torch.max(torch.Tensor([x_mean, 1e-4]))
        x = x + torch.ones_like(x)

        return x.squeeze()
     

class ARL(LightningModule): 

    def __init__(self, input_size=97, output_size=1, batch_size = 256, lr=.001, pretrain_steps=250, betas=None) -> None:
        super(ARL, self).__init__()

        self.learner = nn.Linear(input_size, output_size) 

        self.adversary = AdversaryNN(input_size, output_size)
 

        self.lr = lr

        self.automatic_optimization = False

        self.loss = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task="binary", multiclass=False)
        self.val_acc = Accuracy(task="binary", multiclass=False)
        self.test_acc = Accuracy(task="binary", multiclass=False) 
        self.dp = DemParityMetric() 
        self.dp_train = DemParityMetric()

        self.dp_test = DemParityMetric()
        self.eo_test = EqualOpportunity()
        self.eod_test = EqualizedOdds()

        self.adversary_weights = torch.ones(batch_size)

        self.sigmoid =  nn.Sigmoid()
        self.pretrain_steps = pretrain_steps

    def forward(self, x):
        logits = self.learner(x).squeeze()
        sigmoid_output = self.sigmoid(logits)
        return logits, sigmoid_output

    def learner_loss(self, logits, targets, adversary_weights):
        """
        Compute the learner for the adversary.
        """
        loss =  F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_loss = loss * adversary_weights
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def adv_loss(self, logits, targets, adversary_weights):
        """
        Compute the loss for the adversary.
        """
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') 

        weighted_loss = -(adversary_weights * loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def training_step(self, batch):
        x, y, s = batch 

        optim_learner, optim_adv = self.optimizers()
        

        # Learner Step 
        logits, sig_output = self(x)  
        #with torch.no_grad():
        #    adversary_weights = self.adversary(x)
        #    self.adversary_weights = adversary_weights.detach()

         
        if self.current_epoch > self.pretrain_steps:
            # Adversary Step 
            adversary_weights = self.adversary(x)  
        else:
            adversary_weights = torch.ones(x.shape[0])

        loss = self.learner_loss(logits, y, adversary_weights.detach()) 
        
        self.log('weights', adversary_weights.mean())

        #self.logger.experiment['adversary_weights'].append(adversary_weights)

        
        self.toggle_optimizer(optim_learner, 0) 
        self.manual_backward(loss)
        optim_learner.step()
        optim_learner.zero_grad()
        self.untoggle_optimizer(0) 


        avd_loss = self.adv_loss(logits.detach(), y, adversary_weights)

        if self.current_epoch > self.pretrain_steps:
            self.toggle_optimizer(optim_adv, 1) 
            self.manual_backward(avd_loss) 
            optim_adv.step()
            optim_learner.zero_grad()
            self.untoggle_optimizer(1) 
        
        self.log("arl/adv_loss", avd_loss)
        self.train_acc.update(logits, y.long())
 
        self.log("acc/train", self.train_acc,
                 prog_bar=True, on_epoch=True, on_step=False)
        self.log("arl/leaner_loss", loss) 

        self.dp_train.update(sig_output, s)
        self.log("dp/train", self.dp_train, prog_bar=False,
                 on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, _): 
        x, y, s = batch

        # print(s)
        output, sig_output = self(x)

        loss = self.loss(output, y)

        self.val_acc.update(sig_output, y.long())
 
        self.dp.update(sig_output, s)
        

        self.log("acc/val", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("loss/val", loss, prog_bar=True, on_epoch=True)
        self.log("dp/val", self.dp, prog_bar=True, on_epoch=True)

    def test_step(self, batch, _):
        x, y, s = batch

        _, sigmoid_output  = self.predict(x)

        #loss = self.loss(output, y)

        self.dp_test.update(sigmoid_output, s)
        self.eo_test.update(sigmoid_output, y, s)
        self.eod_test.update(sigmoid_output, y, s)
        self.test_acc.update(sigmoid_output, y.long())

        self.log("acc/test", self.test_acc, on_epoch=True)
        self.log("dp/test", self.dp_test, prog_bar=True, on_epoch=True)
        self.log("eo/test", self.eo_test, prog_bar=True, on_epoch=True)
        self.log("eod/test", self.eod_test, prog_bar=True, on_epoch=True)
        # self.log("test/loss", loss)

    def configure_optimizers(self):
        optim_learner = Adam(self.learner.parameters(), self.lr)
        optim_avd = Adam(self.adversary.parameters(), self.lr)
        return [optim_learner, optim_avd]

    def predict(self, x): 
        _, preds = self(x)
        ans = (preds > .5).long()
        return ans
