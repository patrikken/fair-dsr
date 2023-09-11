from torchmetrics import Metric
import torch
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_negative_rate

class DemParityMetric(Metric):
    """Calculate the demographic parity difference.

    The demographic parity difference is defined as the difference
    between the largest and the smallest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The demographic parity difference of 0 means that all groups have the same selection rate.

    Parameters
    ---------- 

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed 

    Returns
    -------
    float
        The demographic parity difference
    """
    is_differentiable: False
    higher_is_better: False
    full_state_update: bool = True

    def __init__(self, treshold=0.5):
        super().__init__()
        self.add_state("pr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        #self.add_state('dp', default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.treshold = treshold

    def update(self, preds: torch.Tensor, sensitive_features: torch.Tensor):
        y, s = (preds > self.treshold).long(), sensitive_features

        assert preds.shape == sensitive_features.shape

        self.pr = torch.sum(torch.logical_and(y == 1, s == 1)
                             )/torch.sum(s == 1).float()
        self.nr = torch.sum(torch.logical_and(y == 1, s == 0)
                             )/torch.sum(s == 0).float()
        #dp = demographic_parity_difference(y.cpu().numpy(), y.cpu().numpy(), sensitive_features=s.cpu().numpy())
        #self.dp = torch.tensor(dp)
    def compute(self):
        return torch.abs(self.pr - self.nr)


class EqualizedOdds(Metric):
    """Calculate the equalized odds difference.

    The sum of two metrics: `true_positive_rate_difference` and
    `false_positive_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Parameters
    ----------
    preds : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    y_true : array-like
        Ground truth (correct) labels. 

    sensitive_features :
        The sensitive features over which demographic parity should be assessed 

    Returns
    -------
    float
        The equalized odds difference
    """
    is_differentiable: False
    higher_is_better: False
    full_state_update: bool = True

    def __init__(self, treshold=0.5):
        super().__init__()
        self.add_state("tp_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        #self.add_state("result", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.treshold = treshold

    def update(self, preds: torch.Tensor, target: torch.Tensor, sensitive_features: torch.Tensor):
        assert preds.shape == sensitive_features.shape

        y_pred, y_true, s = (
            preds > self.treshold).long(), target, sensitive_features

        self.tp_0 = torch.sum(torch.logical_and(y_pred == 1, torch.logical_and(
            y_true == 1, s == 0))) / torch.sum(torch.logical_and(y_true == 1, s == 0)).float()

        self.tp_1 = torch.sum(torch.logical_and(y_pred == 1, torch.logical_and(y_true == 1, s == 1))) / torch.sum(
            torch.logical_and(y_true == 1, s == 1)).float()

        self.fp_0 = torch.sum(torch.logical_and(y_pred == 1, torch.logical_and(
            y_true == 0, s == 0))) / torch.sum(torch.logical_and(y_true == 0, s == 0)).float()

        self.fp_1 = torch.sum(torch.logical_and(y_pred == 1, torch.logical_and(
            y_true == 0, s == 1))) / torch.sum(torch.logical_and(y_true == 0, s == 1)).float()
        #result = equalized_odds_difference(y_true.numpy(), y_pred.numpy(), sensitive_features=s.numpy())
        #self.result = torch.tensor(result)

    def compute(self):
        return torch.max(torch.abs(self.tp_1 - self.tp_0), torch.abs(self.fp_0 - self.fp_1))


class EqualOpportunity(Metric):
    """Calculate the equal opportunity metric

    The difference between the true_positive_rate_difference :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s).  
    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Parameters
    ----------
    preds : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    y_true : array-like
        Ground truth (correct) labels. 

    sensitive_features :
        The sensitive features over which demographic parity should be assessed 

    Returns
    -------
    float
        The equalized odds difference
    """
    is_differentiable: False
    higher_is_better: False
    full_state_update: bool = True

    def __init__(self, treshold=0.5):
        super().__init__()
        self.add_state("tp_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.treshold = treshold

    def update(self, preds: torch.Tensor, target: torch.Tensor, sensitive_features: torch.Tensor):
        assert preds.shape == sensitive_features.shape

        y_pred, y_true, s = (
            preds > self.treshold).long(), target, sensitive_features

        self.tp_0 = torch.sum(torch.logical_and(y_pred == 1, torch.logical_and(
            y_true == 1, s == 0))) / torch.sum(torch.logical_and(y_true == 1, s == 0)).float()

        self.tp_1 = torch.sum(torch.logical_and(y_pred == 1, torch.logical_and(y_true == 1, s == 1))) / torch.sum(
            torch.logical_and(y_true == 1, s == 1)).float()

    def compute(self):
        return torch.abs(self.tp_1 - self.tp_0)

 

def statistical_parity_score(y_pred, s):
    """ This measure the proportion of positive and negative class in protected and non protected group """

    alpha_1 = np.sum(np.logical_and(y_pred == 1, s == 1)) / \
        float(np.sum(s == 1))
    beta_1 = np.sum(np.logical_and(y_pred == 1, s == 0)) / \
        float(np.sum(s == 0))
    return np.abs(alpha_1 - beta_1)


def equalized_odds(y_true, y_pred, sensitive_features):
    """
        Parameters
        ----------
        y_pred : 1-D array size n
            Label returned by the model
        y_true : 1-D array size n
            Real label
            # print("Training %s"%(name))
        s: 1-D size n protected attribut
        Return
        -------
        equal_opportunity + equal_disadvantage 
        True positive error rate across group + False positive error rate across group
    """
    s = sensitive_features
    alpha_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 0))) / float(
        np.sum(np.logical_and(y_true == 1, s == 0)))
    beta_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 1))) / float(np.sum(
        np.logical_and(y_true == 1, s == 1)))

    alpha_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 0))) / float(np.sum(
        np.logical_and(y_true == 0, s == 0)))
    beta_2 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 1))) / float(np.sum(
        np.logical_and(y_true == 0, s == 1)))

    equal_opportunity = np.abs(alpha_1 - beta_1)
    equal_disadvantage = np.abs(alpha_2 - beta_2)
    return np.max(equal_opportunity, equal_disadvantage)


def equal_opportunity(y_true, y_pred, sensitive_features):
    """
        Parameters
        ----------
        y_pred : 1-D array size n
            Label returned by the model
        y_true : 1-D array size n
            Real label
            # print("Training %s"%(name))
        s: 1-D size n protected attribut
        Return
        -------
        equal_opportunity True positive error rate across group 
    """
    s = sensitive_features

    alpha_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 0))) / float(
        np.sum(np.logical_and(y_true == 1, s == 0)))
    beta_1 = np.sum(np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 1))) / float(np.sum(
        np.logical_and(y_true == 1, s == 1))) 

    equal_opportunity = np.abs(alpha_1 - beta_1) 
    return equal_opportunity