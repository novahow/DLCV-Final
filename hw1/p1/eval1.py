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
# print(args.verbose)
if args.cuda:
    print("cuda:{}".format(args.cuda))

device = ("cuda:" + (args.cuda)) if torch.cuda.is_available() and args.cuda else "cuda"
print("device =", device)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0xC8763)



offset = args.offset


test_tfm = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class CustomDataset(Dataset):
    def __init__(self, offset, mode, transform=None):
        datadir = args.offset
        self.filename = [f for f in listdir(datadir) if isfile(join(datadir, f))]
        
        self.tensor = [0] * len(self.filename)
        self.labels = [0] * len(self.filename)
        self.ids = [0] * len(self.filename)
        for i, name in enumerate(self.filename):
            self.tensor[i] = (Image.open(os.path.join(datadir, name)).convert('RGB'))
            # self.labels[i] = int(name.split('_')[0])
            # self.ids[i] = name
            # self.ids[i] = int(name.split('_')[-1].split('.')[0])
        self.transform = transform
    def __getitem__(self, index):
        x = self.transform(self.tensor[index])
        # label = self.labels[index]
        id = self.filename[index]
        return x, id

    def __len__(self):
        return len(self.labels)

batch_size = 64 # medium: smaller batchsize
learning_rate = 4.5 * 1e-2 if not args.lr else args.lr
eval_batch_size = 32

# train_dataset = CustomDataset(offset, "train", transform=fin_tfm)
val_dataset = CustomDataset(offset, "val", transform=test_tfm)

# train_loader = DataLoader(train_dataset , batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(val_dataset, batch_size=eval_batch_size, pin_memory=True)

# print("datalen = ", len(train_dataset), len(val_dataset))

class SiLU(nn.Module):  # export-friendly version of nn.SiLU() 
    @staticmethod 
    def forward(x): 
        return x * torch.sigmoid(x) 

class Concat(nn.Module):
    def __init__(self, modelA):
        super(Concat, self).__init__()
        self.modelA = modelA
        self.ac = SiLU()
        self.last_layer = nn.Linear(1000, 50)        
    def forward(self, x):
        x = self.modelA(x)
        x = self.ac(x)
        l2_res = x
        x = self.last_layer(x)
        return x, l2_res


from iv4 import inceptionv4
model = inceptionv4()
model = Concat(model).to(device)
# model.load_state_dict(torch.load(os.path.join(offset, 'model2'), map_location=device))

criterion = nn.CrossEntropyLoss()
n_epochs = 100 if not args.n_epoch else args.n_epoch
if args.model:
    model.load_state_dict(torch.load(os.path.join('./', args.model), map_location=device))

lr_factor = 0.94
model.eval()
valid_loss = []
valid_accs = []
best_acc = 0
predictions = []
imageid = []
# Iterate the validation set by batches.
all_valid = torch.zeros(0, 1000).to(device)
all_target = torch.zeros(0)
tot_acc = 0
for batch in (valid_loader):

    # A batch consists of image data and corresponding labels.
    imgs, ids = batch
    # all_target = torch.cat([all_target, labels])
    iid = ids
    # We don't need gradient in validation.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits, res = model(imgs.to(device))
    # We can still compute the loss (but not the gradient).
    # loss = criterion(logits, labels.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    imageid.extend(iid)
    # Compute the accuracy for current batch.
    # bacc = (logits.argmax(dim=-1) == labels.to(device)).float()
    # tot_acc += len(list(filter(lambda e: e == 1, bacc)))
    # acc = bacc.mean()

    # Record the loss and accuracy.
    # valid_loss.append(loss.item())
    # valid_accs.append(acc)

    
# The average loss and accuracy for entire validation set is the average of the recorded values.
# valid_loss = sum(valid_loss) / len(valid_loss)
# valid_acc = (tot_acc) / len(val_dataset)
# print(f"[ Valid | loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

# optimizer.param_groups[0]['lr'] *= lr_factor
'''
model.eval()
# Initialize a list to store the predictions.
predictions = []
imageid = []
# if args.local_rank == 0:
# Iterate the testing set by batches.

for batch in (valid_loader):
    
    imgs, labels, ids = batch
    iid = ['{}_{}.png'.format(labels[i], ids[i]) for i in range(len(imgs))]
    with torch.no_grad():
        logits, res = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    imageid.extend(iid)
'''
# Save predictions into the file.
with open(args.save_dir, "w") as f:

    # The first row must be "Id, Category"
    f.write("image_id,label\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
        f.write(f"{imageid[i]},{pred}\n")