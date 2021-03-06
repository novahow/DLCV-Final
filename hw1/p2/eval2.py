import os
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import random
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
import torchvision.models as models

from torch.optim import Adam, AdamW
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, roc_auc_score
from torchvision.transforms.transforms import ToPILImage
import argparse
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--cuda", help="which cuda?",
                    type=str)
parser.add_argument("--t_epoch", help="tepoch",
                    type=int)
parser.add_argument("--n_epoch", help="nepoch",
                    type=int)
parser.add_argument("--model", help="model",
                    type=str)
parser.add_argument("--lr", help="lr",
                    type=float)
parser.add_argument("--offset", 
                    type=str, default='')
parser.add_argument("--save_dir", 
                    type=str, default='')
args = parser.parse_args()

if args.cuda:
    print("cuda:{}".format(args.cuda))

device = ("cuda:" + (args.cuda)) if torch.cuda.is_available() and args.cuda else "cuda"
print("device =", device)
res = (512, 512)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(123)

offset = args.offset
import torchvision.transforms.functional as TF
class CustomDataset(Dataset):
    def __init__(self, offset, mode, transform=None):
        datadir = args.offset
        self.filename = [f for f in listdir(datadir) if isfile(join(datadir, f))]
        self.satname = [0] * (len(self.filename))
        self.maskname = [0] * (len(self.filename))
        
        for i, name in enumerate(self.filename):
            self.satname[i] = name
        
        self.sat = [0] * len(self.satname)
        self.mask = [0] * len(self.maskname)
        for i in range(len(self.maskname)):
            self.sat[i] = Image.open(join(datadir, self.satname[i])).convert('RGB')
            # self.mask[i] = Image.open(join(datadir, self.maskname[i])).convert('RGB')
        
        self.mode = mode

    def transform(self, image, mask):
        re = transforms.RandomAffine.get_params(degrees=[-10.0, 10.0], translate=None, 
                                            scale_ranges=None, shears=None, img_size=[512, 512])
        image = TF.affine(image, re[0], re[1], re[2], re[3])
        mask = TF.affine(mask, re[0], re[1], re[2], re[3])
        bt = transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.85, 1.15))
        image = bt(image)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        gn = np.random.uniform(0, 0.004, (3, 512, 512))
        image = TF.to_tensor(image)
        image += torch.tensor(gn)
        mask = TF.to_tensor(mask)
        return image, mask
    def __getitem__(self, index):
        x = self.sat[index]
        x = TF.to_tensor(x)
        x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # mks = torch.zeros(512, 512)
        # mask = 4 * label[0, :, :] + 2 * label[1,:,:] + label[2,:,:]
        # mks[mask == 3] = 0  # (Cyan: 011) Urban land 
        # mks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        # mks[mask == 5] = 2  # (Purple: 101) Rangeland 
        # mks[mask == 2] = 3  # (Green: 010) Forest land 
        # mks[mask == 1] = 4  # (Blue: 001) Water 
        # mks[mask == 7] = 5  # (White: 111) Barren land 
        # mks[mask == 0] = 6
        # mks = mks.type(torch.LongTensor)
        return x, self.satname[index]

    def __len__(self):
        return len(self.satname)

learning_rate = 0.0001 if not args.lr else args.lr
eval_batch_size = 8

val_dataset = CustomDataset(offset, "validation", transform=None)
valid_loader = DataLoader(val_dataset, batch_size=eval_batch_size, pin_memory=True)

model = models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=True)

class SiLU(nn.Module):  # export-friendly version of nn.SiLU() 
    @staticmethod 
    def forward(x): 
        return x * torch.sigmoid(x) 

class Concat(nn.Module):
    def __init__(self, modelA):
        super(Concat, self).__init__()
        self.modelA = modelA
        self.bn = nn.BatchNorm2d(21)
        self.ac = SiLU()
        self.last_layer = nn.Conv2d(21, 7, kernel_size=1)      
    def forward(self, x):
        x = self.modelA(x)
        auxloss = x['aux']
        x = x['out']
        x = self.bn(x)
        auxloss = self.bn(auxloss)
        auxloss = self.ac(auxloss)
        x = self.ac(x)
        x = self.last_layer(x)
        auxloss = self.last_layer(auxloss)
        return x, auxloss

# model_name = 'model_rep'
model = Concat(model).to(device)
# model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(join(offset, args.model), map_location=device).items()})
model.load_state_dict(torch.load(join('./', args.model), map_location=device))
model.eval()
# torch.save(model.state_dict(), join(offset, model_name), 
                            # _use_new_zipfile_serialization=False)
# Initialize a list to store the predictions.
predictions = torch.zeros(0, 3, 512, 512)
imageid = []
mbp = [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
pds = torch.zeros(0, 512, 512) 
tot_label = torch.zeros(0, 512, 512) 
from iou import mean_iou_score
for batch in (valid_loader):
    imgs, ids = batch
    # iid = ['{}_pred.png'.format(ids[i].split('_')[0]) for i in range(len(imgs))]
    # tot_label = torch.cat([tot_label, labels])
    with torch.no_grad():
        logits, auxloss = model(imgs.to(device))

    cmap = logits.argmax(dim=1).cpu()
    # pds = torch.cat([pds, cmap], dim=0)
    cmap = cmap.numpy()
    e = torch.zeros(cmap.shape[0], 3, 512, 512)
    for d in range(len(cmap)):
        for r in range(7):
            for c in range(3):
                e[d, c, cmap[d, :] == r] = mbp[r][c]
    # Take the class with greatest logit as prediction and record it.
    predictions = torch.cat([predictions, e], dim=0)
    print(len(imageid) // eval_batch_size)
    imageid.extend(ids)

# iou = mean_iou_score(np.array(pds), np.array(tot_label), dis=False)
# print(f"iou = {iou:.5f}")

# Save predictions into the file.
# predictions = predictions.permute(0, 3, 1, 2)
save_dir = args.save_dir
for i, pred in  enumerate(predictions):
    topil = transforms.ToPILImage()
    img = topil(pred)
    img.save(join(save_dir, imageid[i].split('.')[0] + '.png'))
