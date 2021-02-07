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
import timm
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SamModel:
    def __init__(self, exp_dict):
        if 'resnext' in exp_dict['model']['name']:
            self.network = Resnext(exp_dict)
        elif 'efficientnet' in exp_dict['model']['name']:
            self.network = EfficientNet(exp_dict)
        else:
            self.network = Resnet(exp_dict).network
        self.network.to(DEVICE)
        self.opt = SAM(self.network.parameters(), exp_dict['opt'])
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

            loss.backward()
            self.opt.first_step(zero_grad=True)

            # forward and backward again
            second_logits = self.network(images)
            self.criterion(second_logits, labels).backward()
            self.opt.second_step(zero_grad=True)

            loss_sum += float(loss)
            loss_samples += images.shape[0]

        return {'train_loss': loss_sum / loss_samples}

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

        return {'val_acc': acc_sum / acc_samples}

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


def get_criterion(exp_dict):
    if exp_dict['loss_func']['name'] == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss()

class Resnet():
    def __init__(self, exp_dict):
        self.network = torchvision.models.resnet152()
        self.network.fc = nn.Linear(2048, 5, bias=True)


# ref: https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
class EfficientNet(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        self.model = timm.create_model(exp_config['model']['name'], pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 5)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''

    def forward(self, x):
        x = self.model(x)
        return x


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


class SAM(torch.optim.Optimizer):
    def __init__(self, params, opt_dict):
        opt_dict['weight_decay'] = opt_dict.pop('wd')
        super(SAM, self).__init__(params, opt_dict)

        opt_dict.pop('rho')
        opt_dict.pop('name')
        base_optimizer = torch.optim.SGD
        self.base_optimizer = base_optimizer(self.param_groups, **opt_dict)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
