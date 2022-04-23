import argparse
import os
from unicodedata import normalize
import numpy as np
import math
import os
from os import listdir
from os.path import isfile, join
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, dataloader
from torchvision import datasets
from torch.autograd import Variable
import random

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--cuda", help="which cuda?",
                    type=str, default='0')
parser.add_argument("--t_epoch", help="tepoch",
                    type=int)
parser.add_argument("--n_epoch", help="nepoch",
                    type=int)
parser.add_argument("--model", help="model",
                    type=str)
parser.add_argument("--lr", type=float, 
                    default=0.0002, help="adam: learning rate")
parser.add_argument("--offset", 
                    type=str, default='/tmp2/b08902047/dlcv/hw2/hw2_data/digits/mnistm/gen')
parser.add_argument("--save_dir", 
                    type=str, default='/tmp2/b08902047/dlcv/hw2/hw2_data/digits/mnistm/gen')
parser.add_argument("--model_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/mnistm/ckpts')



args = parser.parse_args()
device = 'cuda:' + args.cuda

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2021)

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

'''from acgan import Generator
generator = Generator().to(device)
generator.load_state_dict(torch.load('./acgan.pt'))

n_gen = 100
for i in range(10):
    
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_gen, args.latent_dim))))
    gen_labels = Variable(LongTensor([i for j in range(n_gen)]))
    gen_imgs = generator(z, gen_labels)
    for j, img in enumerate(gen_imgs):  
        # img = img * 0.5 + 0.5
        img = transforms.Resize(args.img_size)(img)
        torchvision.utils.save_image(img, join(args.save_dir, '{}_{:03d}.png'.format(i, j + 1)), 'png', normalize=True)
'''


transform=transforms.Compose([
                                transforms.Resize(args.img_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.5], [0.5])
                            ]
                        )

# Configure data loader
class CustomDataset(Dataset):
    def __init__(self, offset=None, mode=None, transform=None):
        datadir = join(args.offset, '')
        self.filename = [f for f in listdir(datadir) if isfile(join(datadir, f))]
        # self.labels = df['label']
        # for i, name in enumerate(self.filename):
        #     self.filename[i] = name
        #     self.ids[i] = int(name.split('_')[-1].split('.')[0])
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(join(args.offset, self.filename[index])).convert('RGB')
        label = int(self.filename[index].split('_')[0])
        x = self.transform(img)
        return x, label

    def __len__(self):
        return len(self.filename)

# argsimizers

test_set = CustomDataset(transform=transform)
eval_loader = DataLoader(test_set, 64)


from dc import Classifier
model = Classifier().to(device)
model.load_state_dict(torch.load('./Classifier.pth')['state_dict'])
model.eval()
predictions = []
truth = []

for i, (imgs, labels) in enumerate((eval_loader)):
    imgs = imgs.to(device)
    logit = model(imgs)
    predictions.extend(logit.argmax(dim=-1).cpu().numpy().tolist())
    truth.extend(labels.cpu().numpy().tolist())


print(len(predictions))
print(len(truth))
acc = np.sum(np.array(predictions) == np.array(truth))
print('acc = ', acc / len(predictions))
