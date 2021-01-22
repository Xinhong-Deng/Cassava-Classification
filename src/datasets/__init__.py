from .albunmentation_data import AlbunmentationData
from .torchvision_loader import GetData

import numpy as np 
import pandas as pd
import json
from PIL import Image
import os, tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_data(Dir, FNames, Labels, exp_dict, isTrain=True):
    transform_dict = exp_dict['train_transform'] if isTrain else exp_dict['val_transform']
    
    if transform_dict['name'] in ['tf1', 'tf2', 'tf3']:
        return AlbunmentationData(Dir, FNames, Labels, Transform_dict=transform_dict)
        
    elif transform_dict['name'] == 'default':
       return GetData(Dir, FNames, Labels, Transform_dict=transform_dict)
    else:
        return GetData(Dir, FNames, Labels, Transform_dict={'name': 'default'})


def get_loader(split, exp_dict, datadir='../input/cassava-leaf-disease-classification'):
    
    if split in ['train', 'val']:
        image_dir = f'{datadir}/train_images/'
        labels = json.load(open(f'{datadir}/label_num_to_disease_map.json'))
        train = pd.read_csv(f'{datadir}/train.csv')

        X, Y = train['image_id'].values, train['label'].values

        x_train, x_val, y_train, y_val = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=Y)

        if split == 'val':
            dataset = get_data(image_dir, x_val, y_val, exp_dict, isTrain=False)
            sampler = None 
        else:
            dataset = get_data(image_dir, x_train, y_train, exp_dict, isTrain=True)
            sampler = torch.utils.data.RandomSampler(
                                        dataset, replacement=True, 
                                        num_samples=exp_dict['batch_size']*20)

        loader = DataLoader(dataset, batch_size=exp_dict['batch_size'], sampler=sampler,  num_workers=4)

    elif split == 'test':
        image_dir = f'{datadir}/test_images/'
        X_Test = [name for name in (os.listdir(image_dir))]
        dataset = get_data(image_dir, X_Test, None, exp_dict, isTrain=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return loader