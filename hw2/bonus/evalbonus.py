# import enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from PIL import Image
from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd
# from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy

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
                    default=2e-4, help="adam: learning rate")
parser.add_argument("--offset", 
                    type=str)
parser.add_argument("--save_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')
parser.add_argument("--model_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')
parser.add_argument("--sd", 
                    type=str, default='svhn')
parser.add_argument("--td", 
                    type=str, default='mnistm')

args = parser.parse_args()
device = 'cuda:' + args.cuda
batch_size = 64
print(args)

su = (args.td == 'svhn')
tu = (args.td == 'usps')

img_size = 28 if (su or tu) else 32


t_tfm = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.4], [0.3])
])

class CustomDataset(Dataset):
    def __init__(self, ds=None, mode=None, transform=None):
        self.datadir = args.offset
        self.filename = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))]
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(join(self.datadir, self.filename[index])).convert('RGB')
        x = self.transform(img)
        return x, self.filename[index]

    def __len__(self):
        return len(self.filename)

t_test = CustomDataset(mode='test', ds=args.td, transform=t_tfm)
t_eld = DataLoader(t_test, shuffle=False, batch_size=batch_size)

# print(len(s_train), len(t_train), len(s_rld), len(t_rld))

"""# Model

Feature Extractor: 典型的VGG-like疊法。

Label Predictor / Domain Classifier: MLP到尾。

相信作業寫到這邊大家對以下的Layer都很熟悉，因此不再贅述。
"""

from models.build_gen import *

args.sd = 'usps' if args.td == 'svhn' else args.sd

G = Generator(args.sd, args.td).to(device)
C1 = Classifier(source=args.sd, target=args.td).to(device)
C2 = Classifier(source=args.sd, target=args.td).to(device)

G.load_state_dict(torch.load(join('./bonus', '{}_G.pt'.format(args.td)), map_location=device))
C1.load_state_dict(torch.load(join('./bonus', '{}_C1.pt'.format(args.td)), map_location=device))
C2.load_state_dict(torch.load(join('./bonus', '{}_C1.pt'.format(args.td)), map_location=device))


def ent(output):
        return - torch.mean(output * torch.log(output + 1e-6))


def discrepancy(out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))


num_k = 4
G.eval()
C1.eval()
C2.eval()
preds = []
img_name = []
for i, (imgs, fname) in enumerate((t_eld)):
    # print(imgs.shape, labels.shape)
    img_name.extend(fname)
    imgs = imgs.to(device)
    feat = G(imgs)
    output1 = C1(feat)
    output2 = C2(feat)
    # print(output1.shape)
    output_ensemble = output1 + output2
    preds.extend(torch.argmax(output_ensemble, dim=-1).detach().cpu().numpy())

with open(args.save_dir, "w") as f:

    # The first row must be "Id, Category"
    f.write("image_name,label\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(preds):
        f.write(f"{img_name[i]},{pred}\n")

print('done')
