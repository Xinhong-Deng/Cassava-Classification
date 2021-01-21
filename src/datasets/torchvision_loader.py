from PIL import Image
import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_transform(transform_dict):
    IM_SIZE = 256 if 'im_size' not in transform_dict.keys() else transform_dict['im_size']

    if transform_dict['name'] == 'default':
        return transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((IM_SIZE, IM_SIZE)),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    else:
        return transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((IM_SIZE, IM_SIZE)),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform_dict):
        self.dir = Dir
        self.fnames = FNames
        self.transform = get_transform(Transform_dict)
        self.lbs = Labels
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        if "train" in self.dir:            
            return self.transform(x), self.lbs[index]            
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]