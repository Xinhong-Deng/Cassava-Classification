import numpy as np 
import pandas as pd
import json
from PIL import Image
import os, tqdm
from torch.nn.modules.loss import CrossEntropyLoss

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(exp_dict):
    if 'resnext' in exp_dict['model']['name'] :
        return Resnext(exp_dict)
    elif exp_dict['model']['name'] == 'resnet':
        return Resnet(exp_dict).network
    else:
        return Resnet(exp_dict).network

def get_optimizer(exp_dict, model):
    opt_dict = exp_dict['opt']
    if opt_dict['name'] == 'adamW':
        return torch.optim.AdamW(
            model.parameters(), lr=opt_dict['lr'], weight_decay=opt_dict['wd']
        )
    elif opt_dict['name'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=opt_dict['lr'])
    else:
        return torch.optim.Adam(model.parameters(), lr=opt_dict['lr'])

def get_criterion(exp_dict):
    if exp_dict['loss_func']['name'] == 'symmetric_cross_entropy':
        return SymmetricCrossEntropy(exp_dict)
    elif exp_dict['loss_func']['name'] == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()


class Model:
    def __init__(self, exp_dict):
        self.network = get_model(exp_dict)
        self.network.to(DEVICE)
        self.opt = get_optimizer(exp_dict, self.network)
        self.criterion = get_criterion(exp_dict)

    def train_on_loader(self, loader):
        self.network.train()
        loss_sum = 0.0
        loss_samples = 0.

        for i, (images, labels) in enumerate(tqdm.tqdm(loader, desc='Training')):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = self.network(images)
            loss = self.criterion(logits, labels)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss_sum += float(loss)
            loss_samples += images.shape[0]
            
        return {'train_loss':loss_sum / loss_samples}

    @torch.no_grad()
    def val_on_loader(self, loader):
        self.network.eval()

        acc_sum = 0.0
        acc_samples = 0.
        for (images, labels) in tqdm.tqdm(loader, desc='Validating'): 
            image = images.to(DEVICE)
            
            logits = self.network(image)     
            preds = logits.argmax(dim=1)
            
            acc_sum += float((preds.cpu() == labels).sum())
            acc_samples += labels.shape[0] 

        return {'val_acc': acc_sum/acc_samples }  

    @torch.no_grad()
    def test_on_loader(self, loader):
        self.network.eval()
        s_ls = []

        for image, fname in tqdm.tqdm(loader, desc='Testing'): 
            image = image.to(DEVICE)
            
            logits = self.network(image)        
            preds = logits.argmax(dim=1)
            
            for pred in preds:
                s_ls.append([fname[0], pred.item()])
                    
        sub = pd.DataFrame.from_records(s_ls, columns=['image_id', 'label'])
        sub.head()        

        sub.to_csv("submission.csv", index=False)

class Resnet():
    def __init__(self, exp_dict):
        self.network = torchvision.models.resnet152()
        self.network.fc = nn.Linear(2048, 5, bias=True)


class Resnext(nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        backbone = torch.hub.load(
            "facebookresearch/semi-supervised-ImageNet1K-models", exp_dict['model']['name']
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        in_features = getattr(backbone, "fc").in_features
        self.classifier = nn.Linear(in_features, 5)

        import torch.nn.functional as F
        self.pool_type = F.adaptive_avg_pool2d
    
    def forward(self, x):
        features = self.pool_type(self.backbone(x), 1)
        features = features.view(x.size(0), -1)
        return self.classifier(features)


import torch.nn.functional as F
# ref: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/208239
class SymmetricCrossEntropy(nn.Module):

    def __init__(self, exp_dict):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = exp_dict['loss_func']['alpha']
        self.beta = exp_dict['loss_func']['beta']
        self.num_classes = 5

    def forward(self, logits, targets, reduction='mean'):
        onehot_targets = torch.eye(self.num_classes)[targets].cuda()
        ce_loss = F.cross_entropy(logits, targets, reduction=reduction)
        rce_loss = (-onehot_targets*logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == 'mean':
            rce_loss = rce_loss.mean()
        elif reduction == 'sum':
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss
