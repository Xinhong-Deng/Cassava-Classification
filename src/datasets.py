
import numpy as np 
import pandas as pd
import json
from PIL import Image
import os, tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_loader(split, exp_dict, datadir='../input/cassava-leaf-disease-classification'):
    IM_SIZE = 256

    Transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((IM_SIZE, IM_SIZE)),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    
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
            dataset = GetData(image_dir, x_val, y_val, Transform)
            sampler = None 
        else:
            dataset = GetData(image_dir, x_train, y_train, Transform)
            sampler = torch.utils.data.RandomSampler(
                                        dataset, replacement=True, 
                                        num_samples=exp_dict['batch_size']*20)

        loader = DataLoader(dataset, batch_size=exp_dict['batch_size'], sampler=sampler,  num_workers=4)

    elif split == 'test':
        image_dir = f'{datadir}/test_images/'
        X_Test = [name for name in (os.listdir(image_dir))]
        dataset = GetData(image_dir, X_Test, None, Transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return loader

class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.lbs = Labels
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        if "train" in self.dir:            
            return self.transform(x), self.lbs[index]            
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]
        
