import os
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
import torchvision.models as models
import torchvision
from torch.optim import Adam, AdamW
import pandas as pd
from torch.autograd import Variable
import argparse
from transformers import ViTFeatureExtractor, ViTForImageClassification

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", help="which cuda?",
                    type=str, default='0')
parser.add_argument("--t_epoch", help="tepoch",
                    type=int)
parser.add_argument("--n_epoch", help="nepoch",
                    type=int, default=200)
parser.add_argument("--model", help="model",
                    type=str)
parser.add_argument("--lr", type=float, 
                    default=0.00005, help="adam: learning rate")
parser.add_argument("--offset", 
                    type=str, default='/tmp2/b08902047/dlcv/hw3/hw3_data/p1_data')
parser.add_argument("--img_dir", 
                    type=str, default='/tmp2/b08902047/dlcv/hw3_data')
parser.add_argument("--batch_size", type=int, 
                    default=8, help="bsize")
parser.add_argument("--save_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')
parser.add_argument("--model_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')


args = parser.parse_args()
device = 'cuda:' + args.cuda
img_size = 384

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')


class CustomDataset(Dataset):
    def __init__(self, ds=None, mode=None, transform=None):
        # self.datadir = join(args.offset, mode)
        self.datadir = args.img_dir
        

        self.filename = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))]

        self.labels = [0] * len(self.filename)
        
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(join(self.datadir, self.filename[index])).convert('RGB')
        # img = torchvision.io.read_image(fname)
        x = self.transform(img)
        # print(x.shape)
        x = feature_extractor(images=x, return_tensors='pt')['pixel_values']
        x = torch.squeeze(x, dim=0)
        return x, self.filename[index], self.labels[index]

    def __len__(self):
        return len(self.filename)


test_tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)), 
])

valid_set = CustomDataset(mode='val', transform=test_tfm)
valid_loader = DataLoader(valid_set, shuffle=False, batch_size=args.batch_size, num_workers=2)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384').to(device)
model.num_labels = 37
print(model.classifier)
model.classifier = nn.Linear(768, model.num_labels, bias=True).to(device)

model.load_state_dict(torch.load(join('p1', 'model0.pt'), map_location=device))


model.eval()
preds = []
fname = []
losses = []
for i, (img, _, label) in enumerate((valid_loader)):
    print(i)
    with torch.no_grad():
        img = img.to(device)
        label = label.to(device)
        outputs = model(img, labels=label)            
        logits = outputs.logits
        fname.extend(_)
        pred = logits.argmax(dim=-1).detach().cpu()
        preds.extend(pred)

with open(args.save_dir, "w") as f:

    # The first row must be "Id, Category"
    f.write("filename,label\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(preds):
        f.write(f"{fname[i]},{pred}\n")
