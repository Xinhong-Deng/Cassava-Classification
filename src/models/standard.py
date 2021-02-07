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


class StandardModel:
    def __init__(self, exp_dict):
        if 'resnext' in exp_dict['model']['name']:
            self.network = Resnext(exp_dict)
        elif exp_dict['model']['name'] == 'resnet':
            self.network = Resnet(exp_dict).network
        elif 'efficientnet' in exp_dict['model']['name']:
            self.network = EfficientNet(exp_dict)
        elif 'vit' in exp_dict['model']['name']:
            self.network = ViT(exp_dict)
        elif 'spiralcnn' in exp_dict['model']['name']:
            self.network = SpinalCNN(exp_dict)
        else:
            self.network = Resnet(exp_dict).network
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


def get_optimizer(exp_dict, model):
    opt_dict = exp_dict['opt']
    if opt_dict['name'] == 'adamW':
        return torch.optim.AdamW(
            model.parameters(), lr=opt_dict['lr'], weight_decay=opt_dict['wd']
        )
    elif opt_dict['name'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=opt_dict['lr'])
    elif opt_dict['name'] == 'sam':
        return SAM(model.parameters())
    else:
        return torch.optim.Adam(model.parameters(), lr=opt_dict['lr'])


def get_criterion(exp_dict):
    if exp_dict['loss_func']['name'] == 'symmetric_cross_entropy':
        return SymmetricCrossEntropy(exp_dict)
    elif exp_dict['loss_func']['name'] == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif exp_dict['loss_func']['name'] == 'bitempered':
        return bi_tempered_logistic_loss(exp_dict)
    else:
        return nn.CrossEntropyLoss()


# Networks
# -------


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


class SpinalCNN(nn.Module):
    """CNN."""

    def __init__(self, exp_dict):
        """CNN Builder."""
        super(SpinalCNN, self).__init__()
        self.Half_width = 2048
        self.layer_width = 128
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.Half_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(self.layer_width * 4, 10)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, self.Half_width:2 * self.Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:self.Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, self.Half_width:2 * self.Half_width], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)

        return x


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
        rce_loss = (-onehot_targets * logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == 'mean':
            rce_loss = rce_loss.mean()
        elif reduction == 'sum':
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss


# https://www.kaggle.com/mobassir/vit-pytorch-xla-tpu-for-leaf-disease
class ViT(nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.model = timm.create_model(exp_dict['model']['name'], pretrained=True)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 5)

    def forward(self, x):
        x = self.model(x)
        return x

      
class bi_tempered_logistic_loss(nn.Module):
    """Bi-Tempered Logistic Loss.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations (onehot),
        or a long tensor of one dimension less than activations (pytorch standard)
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      label_smoothing: Label smoothing parameter between [0, 1). Default 0.0.
      num_iters: Number of iterations to run the method. Default 5.
      reduction: ``'none'`` | ``'mean'`` | ``'sum'``. Default ``'mean'``.
        ``'none'``: No reduction is applied, return shape is shape of
        activations without the last dimension.
        ``'mean'``: Loss is averaged over minibatch. Return shape (1,)
        ``'sum'``: Loss is summed over minibatch. Return shape (1,)
    Returns:
      A loss tensor.
    """

    def __init__(self, exp_dict):
        super(bi_tempered_logistic_loss, self).__init__()
        self.t1 = exp_dict['loss_func']['t1']
        self.t2 = exp_dict['loss_func']['t2']
        self.num_iters = 5

        self.reduction = exp_dict['loss_func']['reduction']

    def forward(self, activations, labels, reduction='mean'):
        if len(labels.shape) < len(activations.shape):  # not one-hot
            labels_onehot = torch.zeros_like(activations)
            labels_onehot.scatter_(1, labels[..., None], 1)
        else:
            labels_onehot = labels

        if self.label_smoothing > 0:
            num_classes = labels_onehot.shape[-1]
            labels_onehot = (1 - self.label_smoothing * num_classes / (num_classes - 1)) \
                            * labels_onehot + \
                            self.label_smoothing / (num_classes - 1)

        probabilities = tempered_softmax(activations, self.t2, self.num_iters)

        loss_values = labels_onehot * log_t(labels_onehot + 1e-10, self.t1) \
                      - labels_onehot * log_t(probabilities, self.t1) \
                      - labels_onehot.pow(2.0 - self.t1) / (2.0 - self.t1) \
                      + probabilities.pow(2.0 - self.t1) / (2.0 - self.t1)
        loss_values = loss_values.sum(dim=-1)  # sum over classes

        if reduction == 'none':
            return loss_values
        if reduction == 'sum':
            return loss_values.sum()
        if reduction == 'mean':
            return loss_values.mean()


def log_t(u, t):
    """Compute log_t for `u'."""
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u'."""
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters):
    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0

    for _ in range(num_iters):
        logt_partition = torch.sum(
            exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * \
                                 logt_partition.pow(1.0 - t)

    logt_partition = torch.sum(
        exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = - log_t(1.0 / logt_partition, t) + mu

    return normalization_constants


def compute_normalization_binary_search(activations, t, num_iters):
    """Returns the normalization value for each example (t < 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (< 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu

    effective_dim = \
        torch.sum(
            (normalized_activations > -1.0 / (1.0 - t)).to(torch.int32),
            dim=-1, keepdim=True).to(activations.dtype)

    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(shape_partition, dtype=activations.dtype, device=activations.device)
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)

    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(
            exp_t(normalized_activations - logt_partition, t),
            dim=-1, keepdim=True)
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
            lower * update + (1.0 - update) * logt_partition,
            shape_partition)
        upper = torch.reshape(
            upper * (1.0 - update) + update * logt_partition,
            shape_partition)

    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    """
    Class implementing custom backward pass for compute_normalization. See compute_normalization.
    """

    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(activations, t, num_iters)
        else:
            normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output

        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Backward pass is implemented.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_sigmoid(activations, t, num_iters=5):
    """Tempered sigmoid function.
    Args:
      activations: Activations for the positive class for binary classification.
      t: Temperature tensor > 0.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    internal_activations = torch.stack([activations,
                                        torch.zeros_like(activations)],
                                       dim=-1)
    internal_probabilities = tempered_softmax(internal_activations, t, num_iters)
    return internal_probabilities[..., 0]


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A probabilities tensor.
    """
    if t == 1.0:
        return activations.softmax(dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, rho=0.05, **kwargs):
        base_optimizer = torch.optim.SGD
        default = dict(rho = rho, **kwargs)
        super(SAM, self).__init__(params, default)

        self.base_optimizer = base_optimizer
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
