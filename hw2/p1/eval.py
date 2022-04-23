import os
from os import listdir
from os.path import isfile, join
import numpy as np
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
import torchvision
from torch.optim import Adam, AdamW
import pandas as pd
from torch.autograd import Variable
from torchvision.transforms.transforms import ToPILImage
import argparse

from math import log2
parser = argparse.ArgumentParser()

parser.add_argument("--cuda", help="which cuda?",
                    type=str, default='0')
parser.add_argument("--t_epoch", help="tepoch",
                    type=int)
parser.add_argument("--n_epoch", help="nepoch",
                    type=int)
parser.add_argument("--model", help="model",
                    type=str)
parser.add_argument("--lr", help="lr",
                    type=float)
parser.add_argument("--offset", 
                    type=str, default='/tmp2/b08902047/dlcv/hw2/hw2_data/face/ckpts/default')
parser.add_argument("--save_dir", 
                    type=str, default='/tmp2/b08902047/dlcv/hw2/hw2_data/face/gen/res')
args = parser.parse_args()

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


device = 'cuda:' + args.cuda
from st2 import StyleGAN2
model = StyleGAN2(64, 512, device=device, attn_layers=[1, 2])
# print(model.state_dict())
# gmod = torch.load(join(args.offset, 'model_{}.pt'.format(args.model)), map_location=device)
# print(gmod)
model.load_state_dict(torch.load(join('p1', 'style.pt'), map_location=device)['GAN'], strict=False)

from req_f import Uvar
uv = Uvar(device)

@torch.no_grad()
def evaluate(num = 0, trunc = 1.0):
    model.eval()
    ext = 'png'
    num_rows = 1

    latent_dim = 512
    image_size = 64
    num_layers = int(log2(64) - 1)

    # latents and noise

    latents = uv.noise_list(num_rows ** 2, num_layers, latent_dim)
    n = uv.image_noise(num_rows ** 2, image_size)

    # regular
    model.S.eval()
    model.G.eval()
    # print(model.G)
    generated_images = uv.generate_truncated(model.S, model.G, latents, n, trunc_psi = 0.9)
    torchvision.utils.save_image(generated_images, join(args.save_dir, '{:04d}.{}'.format(num, ext)), nrow=num_rows)


for i in range(1, 1001):
    print(i)
    evaluate(i)
    
