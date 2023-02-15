import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import random
import torchvision.transforms.functional as F
class ISICDataset(Dataset):
    def __init__(self, data_path, csv_file, img_folder, transform = None, training = True, flip_p=0.5):
        df = pd.read_csv(os.path.join(data_path, csv_file), encoding='gbk')
        self.img_folder = img_folder
        self.name_list = df.iloc[:,0].tolist()
        self.label_list = df.iloc[:,1].tolist()
        self.data_path = data_path
        self.transform = transform
        self.training = training
        self.flip_p = flip_p
    def __len__(self):
        return len(self.name_list)
    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]+'.jpg'
        img_path = os.path.join(self.data_path, self.img_folder, name)

        mask_name = name.split('.')[0] + '_Segmentation.png'
        msk_path = os.path.join(self.data_path, self.img_folder, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        if self.training:
            label = 0 if self.label_list[index] == 'benign' else 1
        else:
            label = int(self.label_list[index])

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            if random.random() < self.flip_p:
                img = F.vflip(img)
                mask = F.vflip(mask)


        if self.training:
            return (img, mask)
        return (img, mask, label)
