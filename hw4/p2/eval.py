import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from os.path import join, isfile
from os import listdir
import torch.nn as nn
import sklearn.metrics as metrics
# test model, a resnet 50

resnet = models.resnet50(pretrained=False)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
parser.add_argument('--load', type=str, help="Model checkpoint path")
parser.add_argument('--test_csv', type=str, help="Testing images csv file")
parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
parser.add_argument('--testcase_csv', type=str, help="Test case csv")
parser.add_argument('--output_csv', type=str, help="Output filename")
parser.add_argument('--offset', type=str, help="offset")
parser.add_argument('--cuda', type=int, help="cuda", default=0)
parser.add_argument('--ipc', type=int, help="ipc", default=600)
parser.add_argument('--n_epoch', type=int, help="epochs", default=100)



# constants

BATCH_SIZE = 8
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 1
IMAGE_SIZE = 384
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

import torch
from torchvision import models






filenameToPILImage = lambda x: Image.open(x).convert('RGB')

stfm = transforms.Compose([
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
    transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.0)),
])

from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

atfm = transforms.RandomChoice([
    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
])

train_tfm = transforms.Compose([
        transforms.RandomOrder([
            transforms.RandomChoice([atfm, stfm]),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class OfficeDataset(Dataset):
    def __init__(self, csv_path=None, data_dir=None, mode=None, transform=None):
        
        csv_path = args.test_csv
        self.data_dir = args.offset
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transform
        # self.images = [filenameToPILImage(join(self.data_dir, self.data_df.loc[i, 'filename'])) for i in range(len(self.data_df))]
    def __getitem__(self, index):
        # print(index // 600)
        # path = self.data_df.loc[index, "filename"]
        fn = self.data_df.loc[index, 'filename']
        image = self.transform(filenameToPILImage(join(self.data_dir, fn)))
        # print(image.shape)
        # print(path, label)
        return image, self.data_df.loc[index, 'filename']

    def __len__(self):
        return len(self.data_df)

def encode(file):
    mp = {}
    data_df = pd.read_csv(file).set_index("id")
    cls = np.unique(data_df['label'].values)
    ccnt = 0
    for e in cls:
        mp[ccnt] = e
        ccnt += 1 

    return mp




if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda:{}'.format(args.cuda)
    val_set = OfficeDataset(data_dir=args.offset, mode='val', transform=test_tfm)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
    back = resnet
    

    from Bonemlp import MLP
    model = MLP(back, 65).to(device)

    
    model.load_state_dict(torch.load(join('p2', 'ssl0.pt'), map_location=device))
    best_loss = 1e9
    best_acc = 0
    model.eval()
    preds = []
    fname = []
    with torch.no_grad():
        for img, label in (val_loader):
            x = img.to(device)
            logits = model(x)
            pred = (torch.argmax(logits, dim=-1).cpu().numpy())
            preds.extend(pred)
            fname.extend(label)

    
    idx = encode(join('p2', 'val.csv'))
    with open(args.output_csv, "w") as f:

        # The first row must be "Id, Category"
        f.write("id,filename,label\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(preds):
            f.write(f"{i},{fname[i]},{idx[pred]}\n")

        