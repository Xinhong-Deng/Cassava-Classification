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
from .standard import StandardModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(exp_dict):
    if exp_dict['model']['name'] in ['resnet', 'resnext', 'resnext50_32x4d_ssl']:
        return StandardModel(exp_dict)
    elif exp_dict['model']['name'] in ['clip']:
        return ClipModel(exp_dict)
    else:
        raise ValueError(f'{exp_dict["model"]["name"]} is not available...')

