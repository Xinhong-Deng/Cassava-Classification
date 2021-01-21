import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from torch.utils.data import Dataset
import cv2

def get_transform(transform_dict):
    IM_SIZE = 256 if 'im_size' not in transform_dict.keys() else transform_dict['im_size']

    if transform_dict['name'] == 'tf1':
        # https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
        return A.Compose([
            A.RandomResizedCrop(IM_SIZE, IM_SIZE),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
    elif transform_dict['name'] == 'tf2':
        # https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
        return A.Compose([
            A.CenterCrop(IM_SIZE, IM_SIZE, p=1.),
            A.Resize(IM_SIZE, IM_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
    elif transform_dict['name'] == 'tf3':
        # https://www.kaggle.com/debarshichanda/cassava-bitempered-logistic-loss
        return A.Compose([
        A.RandomResizedCrop(IM_SIZE, IM_SIZE),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        A.CoarseDropout(p=0.5),
        A.Cutout(p=0.5),
        ToTensorV2()], p=1.)
    else:
        return A.Compose([
            A.CenterCrop(IM_SIZE, IM_SIZE, p=1.),
            A.Resize(IM_SIZE, IM_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
    

class AlbunmentationData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform_dict):
        self.dir = Dir
        self.fnames = FNames
        self.transform = get_transform(Transform_dict)
        self.lbs = Labels
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = cv2.imread(os.path.join(self.dir, self.fnames[index]))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if "train" in self.dir:
            if self.transform:
                x = self.transform(image=x)            
            return x['image'], self.lbs[index]            
        elif "test" in self.dir:    
            if self.transform:
                x = self.transform(image=x)   
            return x['image'], self.fnames[index]
