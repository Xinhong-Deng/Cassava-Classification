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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, exp_dict):
        self.network = torchvision.models.resnet152()
        self.network.fc = nn.Linear(2048, 5, bias=True)
        self.network.to(DEVICE)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=exp_dict['lr'])

    def train_on_loader(self, loader):
        self.network.train()
        loss_sum = 0.0
        loss_samples = 0.

        for i, (images, labels) in enumerate(tqdm.tqdm(loader, desc='Training')):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = self.network(images)
            loss = nn.CrossEntropyLoss()(logits, labels)

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