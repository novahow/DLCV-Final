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
import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
from transformers import ViTFeatureExtractor, ViTForImageClassification
from tqdm.auto import tqdm
from aug import ImageNetPolicy, CIFAR10Policy, SVHNPolicy

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
                    type=str, default='/tmp2/b08902047/dlcv/hw3_data/val')
parser.add_argument("--batch_size", type=int, 
                    default=64, help="bsize")
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
        self.datadir = join(args.offset, mode)
        # self.datadir = args.offset
        # df = pd.read_csv(join(self.dm, '{}.csv'.format(mode)))

        self.filename = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))]

        self.labels = [0] * len(self.filename)
        for i, name in enumerate(self.filename):
            # print(name)
            self.labels[i] = int(name.split('_')[0])
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


ntfm = transforms.Compose([
    transforms.RandomOrder([
        transforms.RandomAffine(10, translate=(0.1, 0.1)),
        transforms.ColorJitter(0.15, 0.1, 0.1, 0.1),
        transforms.RandomGrayscale(0.25),
    ])
])

atfm = transforms.Compose([
    transforms.RandomChoice([ImageNetPolicy(), CIFAR10Policy(), SVHNPolicy()]),

])

tfm = transforms.Compose([
    transforms.RandomChoice([atfm, ntfm]),
    transforms.RandomResizedCrop((img_size, img_size)), 
    # transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])

test_tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)), 
    # transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])

train_set = CustomDataset(mode='train', transform=tfm)
valid_set = CustomDataset(mode='val', transform=test_tfm)

train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=4)
valid_loader = DataLoader(valid_set, shuffle=False, batch_size=args.batch_size, num_workers=2)

# all_img = torch.cat([torch.unsqueeze(x[0], 0) for x in train_set], dim=0)
# print(torch.mean(all_img, dim=[0, 2, 3]), torch.std(all_img, dim=[0, 2, 3]))

# feature_extractor.do_resize = False
# feature_extractor.do_normalize = False

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384').to(device)
model.num_labels = 37
print(model.classifier)
model.classifier = nn.Linear(768, model.num_labels, bias=True).to(device)
# model.classifier = nn.Linear(1024, model.num_labels, bias=True).to(device)

'''feature_extractor = ViTFeatureExtractor(do_resize=False,
        size=384,
        resample=Image.BILINEAR,
        do_normalize=False,
        image_mean=None,
        image_std=None,
)

class Config:
    def __init__(self):
        self.num_labels = 37
        self.hidden_size = 768

config = Config()
model = ViTForImageClassification(
).to(device)
model.num_labels = 37
print(model.classifier)
model.classifier = nn.Linear(768, model.num_labels, bias=True).to(device)
'''
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

accum_iter = 4
criterion = nn.CrossEntropyLoss()
ev = 0
best_acc = 0
for epoch in range(args.n_epoch):
    losses = []
    preds = []
    truth = []
    model.train()
    if not ev:
        for i, (img, _, label) in enumerate(tqdm(train_loader)):

            img = img.to(device)
            # inputs = feature_extractor(images=img)
            label = label.to(device)
            outputs = model(pixel_values=img, labels=label)
            # outputs = model(**inputs, labels=label)
            logits = outputs.logits
            # print(logits.shape)
            loss = outputs.loss / accum_iter
            loss.backward()

            if (i + 1) % accum_iter == 0:
                optimizer.step()                    
                optimizer.zero_grad()      
            # optimizer.zero_grad()
            losses.append(loss.detach().cpu().item())
            pred = logits.argmax(dim=-1).detach().cpu()
            # optimizer.step()
            preds.extend(pred)
            
            truth.extend(label.detach().cpu().numpy())
        # print(preds, len(preds))
        # print(truth, len(truth))  
        acc = (np.array(preds) == np.array(truth)).mean()
        
        print('train: epoch:{}/{}, loss = {}, acc = {}'.format(epoch + 1, args.n_epoch, np.mean(losses), acc))    
    
    ev = 0
    model.eval()
    preds = []
    truth = []
    losses = []
    for i, (img, _, label) in enumerate(tqdm(valid_loader)):
        # print(img.shape)
        # 
        with torch.no_grad():
            img = img.to(device)
            # inputs = feature_extractor(images=img, return_tensors="pt")
            label = label.to(device)
            outputs = model(img, labels=label)            
            # outputs = model(**inputs, labels=label)

            logits = outputs.logits
            loss = outputs.loss
            losses.append(loss.detach().cpu().item())
            
            pred = logits.argmax(dim=-1).detach().cpu()
            # print(pred.shape)
            preds.extend(pred)
            truth.extend(label.cpu().numpy())

    # print(preds, len(preds))
    # print(truth, len(truth))
    acc = (np.array(preds) == np.array(truth)).mean()
    print('valid: epoch:{}/{}, loss = {}, acc = {}'.format(epoch + 1, args.n_epoch, np.mean(losses), acc))

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), join(args.offset, 'model0.pt'))