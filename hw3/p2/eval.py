import torch

from transformers import BertTokenizer
from PIL import Image
import argparse
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import os
from os import listdir
from os.path import isfile, join

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import numpy as np
import matplotlib.pyplot as plt
import random
parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--save_dir', type=str, help='path to savedir', required=True)

args = parser.parse_args()
version = 'v3'
device = 'cuda:0'
config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True).to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

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

same_seeds(42)

class CustomDataset(Dataset):
    def __init__(self, ds=None, mode=None, transform=None):
        # self.datadir = join(args.offset, mode)
        self.datadir = args.path
        

        self.filename = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))]

        self.labels = [0] * len(self.filename)
        
        self.transform = coco.val_transform
    def __getitem__(self, index):
        img = Image.open(join(self.datadir, self.filename[index])).convert('RGB')
        # img = torchvision.io.read_image(fname)
        x = self.transform(img)
        # print(x.shape)
        x = torch.unsqueeze(x, dim=0)
        print(x.shape)
        return x, self.filename[index], self.labels[index]

    def __len__(self):
        return len(self.filename)


val_dataset = CustomDataset()

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long).to(device)
    mask_template = torch.ones((1, max_length), dtype=torch.bool).to(device)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False
    
    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

activation = {}
model.eval()
@torch.no_grad()
def get_activation(name):
    def hook(model, input, output):
        # print(output[0].shape, output[1].shape, len(output))
        activation[name] = output[-1].detach()
    return hook
model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(get_activation('multihead_attn'))

@torch.no_grad()
def evaluate(image):
    res = None
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image.to(device), caption, cap_mask)
        res = activation['multihead_attn'].squeeze(0)
        print(res.shape, image.shape)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption, res
        # print(i, tokenizer.decode(caption[0].tolist()))
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption, res





@torch.no_grad()
def plot_attention(image, result, attention_plot):
    temp_image = image[0]

    fig = plt.figure(figsize=(20, 20))

    len_result = len(result)
    for i in range(len_result):
        # h = int(pow(attention_plot[i].shape[-1], 0.5)) + 1
        # w = attention_plot[i].shape[-1] // h + 1
        # h = max(h, w)
        # w = h
        
        
        grid_size = int(max(np.ceil(len_result/2), 2))
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        ax.axis('off')
        img = ax.imshow(temp_image.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5)
        if i:
          h = attention_plot[i - 1].shape[-1] // 19
          temp_att = np.resize(attention_plot[i - 1], (h, 19))
          ax.imshow(temp_att, cmap=plt.cm.rainbow, alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(join(args.save_dir, image[1].split('.')[0] + '.png'))
    # plt.show()


for e in val_dataset:
    caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)
    model.eval()
    output, atm = evaluate(e[0])
    
    # print()
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(result.capitalize())
    result = ['<start>'] + result.split() + ['<end>']
    plot_attention(e, result, atm[:len(result) - 1].cpu())