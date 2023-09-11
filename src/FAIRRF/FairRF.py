import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pandas.core.frame import DataFrame

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from metrics import equal_opportunity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_negative_rate
import utils 
import argparse
from FAIRRF.datasets import get_old_adult, get_adult, get_compas_race, get_compas, get_lsac, get_lsac_sex
import os 
parser = argparse.ArgumentParser(description='FairML')
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--pretrain_epoch", default=1, type=int)
parser.add_argument("--method", default="learnCorre", type=str,choices=['base','corre','groupTPR','learn','remove','learnCorre'])
parser.add_argument("--dataset", default="adult", type=str, choices=['adult','lsac_sex', 'compas_race','new_adult'])
parser.add_argument("--s", default="sex", type=str) #sex for adult
parser.add_argument("--related",nargs='+', type=str)#choices=['sex','race','age','relationship','marital-status', 'education', 'workclass'] for adult
parser.add_argument("--r_weight",nargs='+', type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--weightSum", default=0.3, type=float)# used for learning related weights, weight for corre attr
parser.add_argument("--beta", default=0.5, type=float)#weight for regularization of Lambda

parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model", default='LR', type=str, choices=['MLP', 'LR', 'SVM'])#weight for regularization of Lambda

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print('beta: {}, weightSum: {}'.format(args.beta, args.weightSum))

#-----------------------
#data analysis
#-----------------------
#load data

datasets = {
    'adult': get_old_adult,
    'new_adult': get_adult,
    'compas_race': get_compas_race,
    'compas':get_compas,
    'lsac': get_lsac, 
    'lsac_sex':get_lsac_sex
}

datasets_csv = {
    'adult': 'preprocessing/adult.data1.csv',
    'new_adult': 'data/sampled_new_adult.csv',
    'compas_race': 'preprocessing/compas.csv',
    'compas':'preprocessing/compas.csv',
    'lsac': 'preprocessing/lsac.data1.csv', 
    'lsac_sex':'preprocessing/lsac.data1.csv'
}

fair_metrics_map = {
    'dp': demographic_parity_difference, 
    'eop': equal_opportunity,
    'eodds': equalized_odds_difference
}


data1, data2 = datasets[args.dataset]()

X, y_true, sensitive_attr, processed_X_train = data1 

print(processed_X_train.shape)

#-----------------------
#preprocessing
#-----------------------
# split into train/test set
indict = np.arange(sensitive_attr.shape[0])
(X_train, X_test, y_train, y_test, ind_train, ind_test) = train_test_split(X, y_true, indict, test_size=0.5,
                                      random_state=7)
s_test = sensitive_attr[ind_test]
# standardize the data
#processed_X_train = X_train
#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, np.ndarray):
            return torch.from_numpy(df).float()
        return torch.from_numpy(df.values).float()


train_data = utils.DatasetLoader(torch.from_numpy(X_train), torch.from_numpy(y_train).long(), None, transform=None)  #PandasDataSet(X_train, y_train, ind_train)
test_data = PandasDataSet(X_test, y_test, ind_test)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True, drop_last=True)

print('# training samples:', len(train_data))
print('# batches:', len(train_loader))

#-----------------------
#model
#-----------------------
class Classifier(nn.Module):

    def __init__(self, n_features, n_class=2, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden*2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, x):
        return self.network(x)

class Classifier_lr(nn.Module):
    def __init__(self, n_features, n_class=2):
        super(Classifier_lr, self).__init__()

        self.linear = nn.Linear(n_features, n_class)


    def forward(self, x): 
        return self.linear(x)

def loss_SVM(result, truth, model):
    truth[truth==0] = -1
    result = result.squeeze()
    weight = model.linear.weight.squeeze()

    loss = torch.mean(torch.clamp(1 - truth * result, min=0))
    loss += 0.1*torch.mean(torch.mul(weight, weight))

    return loss


n_features = X.shape[1]
#print('feature dimension: {}'.format(n_features))
if args.dataset=='pokec':
    n_hid = 72
else:
    n_hid = 32

n_classes = y_true.max()+1
clf = Classifier_lr(n_features=n_features,n_class=n_classes)

clf_optimizer = optim.Adam(clf.parameters(), lr=args.lr)

#-----------------------
#run
#-----------------------
##Baseline
''
def pretrain_classifier(clf, data_loader, optimizer, criterion):
    for x, y  in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM': 
            loss = criterion(p_y, y.long().squeeze())
        else:
            loss = criterion(p_y, y, clf)
        loss.backward()
        optimizer.step()
    return clf

##feature-pertubation loss
def Perturb_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        
        for related_attr, related_weight in zip(related_attrs, related_weights):
            #x_new = utils.counter_sample(data.data.iloc[ind.int()], related_attr, scaler)
            x_new = utils.counter_sample(data.data, ind.int(), related_attr, scaler)
            p_y_new = clf(x_new)

            #cor_loss = torch.square(p_y[:,1] - p_y_new[:,1]).mean()
            p_stack = torch.stack((p_y[:,1], p_y_new[:,1]), dim=1)
            p_order = torch.argsort(p_stack,dim=-1)
            cor_loss = torch.square(p_stack[:,p_order[:,1].detach()] - p_stack[:,p_order[:,0]]).mean()

            #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss + cor_loss*related_weight

        loss.backward()
        optimizer.step()

    return clf

def CorreErase_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)

        for related_attr, related_weight in zip(related_attrs, related_weights):
            selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
            cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

            #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
            loss = loss + cor_loss*related_weight

        loss.backward()
        optimizer.step()

    return clf

##group fairness loss
def Gfair_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights):
    for x, y, ind in data_loader:
        clf.zero_grad()
        p_y = clf(x)
        if args.model != 'SVM':
            loss = criterion(p_y, y.long())
        else:
            loss = criterion(p_y, y, clf)
        #
        for related_attr, related_weight in zip(related_attrs, related_weights):
            group_TPR = utils.groupTPR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
            #group_TPR_loss = (group_TPR - (sum(group_TPR)/len(group_TPR)).detach()).sum()*related_weight
            group_TPR_loss = torch.square(max(group_TPR).detach() - min(group_TPR))

            #group_TNR = utils.groupTNR(p_y, y, np.array(data_frame[related_attr].tolist()), ind)
            #group_TNR_loss = torch.square(max(group_TNR).detach() - min(group_TNR))
            #print('classification loss: {}, group TPR loss: {}, group TNR loss: {}'.format(loss.item(), group_TPR_loss.item(), group_TNR_loss.item()))
            #print('classification loss: {}, group TPR loss: {}'.format(loss.item(), group_TPR_loss.item()))
            loss = loss + group_TPR_loss*related_weight
        loss.backward()
        optimizer.step()
    return clf

#correlation regularization with learned weights
def CorreLearn_train(clf, data_loader, optimizer, criterion, related_attrs, related_weights, weightSum):
    
    for x, y in data_loader:
        UPDATE_MODEL_ITERS = 1
        UPDATE_WEIGHT_ITERS = 1

        #update model
        for iter in range(UPDATE_MODEL_ITERS):
            clf.zero_grad()
            p_y = clf(x)
            if args.model != 'SVM': 
                loss = criterion(p_y, y.squeeze().long())
            else:
                loss = criterion(p_y, y, clf)

            for related_attr, related_weight in zip(related_attrs, related_weights.tolist()):
                selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
                cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

                #print('classification loss: {}, feature correlation loss: {}'.format(loss.item(), cor_loss.item()))
                loss = loss + cor_loss*related_weight*weightSum

            loss.backward()
            optimizer.step()

        #update weights
        #ipdb.set_trace()
        for iter in range(UPDATE_WEIGHT_ITERS):
            with torch.no_grad():
                p_y = clf(x)

                cor_losses = []
                for related_attr in related_attrs:
                    selected_column = list(map(lambda s,related_attr: related_attr in s, processed_X_train.keys(),[related_attr]*len(processed_X_train.keys())))
                    cor_loss = torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1,x.shape[0],-1) - x[:,selected_column].mean(dim=0).reshape(1,1,-1), (p_y-p_y.mean(dim=0)).transpose(0,1).reshape((-1,p_y.shape[0],1))),dim=1)))

                    cor_losses.append(cor_loss.item())

                cor_losses = np.array(cor_losses)

                cor_order = np.argsort(cor_losses)

                #compute -v. represent it as v.
                beta = args.beta
                v = cor_losses[cor_order[0]]+ 2*beta
                cor_sum = cor_losses[cor_order[0]]
                l=1
                for i in range(cor_order.shape[0]-1):
                    if cor_losses[cor_order[i+1]] < v:
                        cor_sum = cor_sum + cor_losses[cor_order[i+1]]
                        v = (cor_sum+2*beta)/(i+2)
                        l = l+1
                    else:
                        break
                
                #compute lambda
                for i in range(cor_order.shape[0]):
                    if i <l:
                        related_weights[cor_order[i]] = (v-cor_losses[cor_order[i]])/(2*beta)
                    else:
                        related_weights[cor_order[i]] = 0



                '''
                #older optimization version
                #update
                #related_weights = related_weights - cor_losses*0.001
                #mapping
                #related_weights[related_weights<0] = 0
                #related_weights = related_weights/sum(related_weights)*weightSum
                '''


    return clf, related_weights

#train
related_attrs = args.related
print('Relation >>>>>>>>>>> ', args.related)
related_weights = args.r_weight

if args.model != 'SVM':
    clf_criterion = nn.CrossEntropyLoss()
else:
    clf_criterion = loss_SVM

for i in range(args.pretrain_epoch):
    clf = clf.train()
    clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)


for epoch in range(args.epoch):
    print("================== epoch {} =======================".format(epoch))
    clf = clf.train()
    if args.method == 'base':
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)
    if args.method == 'conterfactual':#only implemented for dataset ADULT
        clf = Perturb_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    if args.method == 'corre':
        clf = CorreErase_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    if args.method == 'groupTPR':
        clf = Gfair_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights)
    if args.method == 'learnCorre':
        related_weights = np.array(related_weights)
        clf, related_weights = CorreLearn_train(clf, train_loader, clf_optimizer, clf_criterion, related_attrs, related_weights, args.weightSum)



#test
clf = clf.eval()
with torch.no_grad():
    pre_clf_test = clf(test_data.tensors[0])

if args.model != 'SVM':
    y_pred = pre_clf_test.argmax(dim=1).numpy()
else:
    y_pred = (pre_clf_test>0).reshape(-1).int()

print('sensitive attributes: ') 
#print('labels: ')
#print(set(y_test))
#print('label dist: ')
#unique, counts = np.unique(y_test, return_counts=True)
#print(dict(zip(unique, counts)))

print('sum of weights weightSUM for learning: {}'.format(args.weightSum))
print('learned lambdas: {}'.format(related_weights))

results = {}
results['acc'] = accuracy_score(y_test, y_pred)

for fair_metric in fair_metrics_map:
    results[fair_metric] = [fair_metrics_map[fair_metric](y_test, y_pred, sensitive_features=s_test)]

idx_0 = np.where(s_test==0)
idx_1 = np.where(s_test==1)

results['acc_0'] = [accuracy_score(y_test[idx_0], y_pred[idx_0])]
results['acc_1'] = [accuracy_score(y_test[idx_1], y_pred[idx_1])]
results['count_0'] = [len(y_test[idx_0])/len(y_test)]
results['count_1'] = [len(y_test[idx_1])/len(y_test)]

print(results)

out_path = 'analysis/results/baselines/{}'.format("FAIRRF",
                                            args.dataset)
out_file = '{}/{}.csv'.format(out_path, args.dataset)
print(out_file)

os.makedirs(out_path, exist_ok=True)
results.update({'seed': args.seed})
if args.seed == 0:
    df = pd.DataFrame(results)
    df.to_csv(out_file, encoding='utf-8', index=False)
else:
    df = pd.read_csv(out_file, index_col=False)
    df = pd.concat([df, pd.DataFrame(results)], axis=0)
    df.to_csv(out_file, encoding='utf-8', index=False) 