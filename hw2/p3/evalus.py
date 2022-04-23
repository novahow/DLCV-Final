import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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
                    default=0.0002, help="adam: learning rate")
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
args.model_dir = join(join(args.offset, args.sd), 'ckpts')
device = 'cuda:' + args.cuda
batch_size = 64
print(args)
img_size = 28

su = (args.sd == 'usps')
tu = (args.td == 'usps')
tfm = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomGrayscale(p=1 if (su or tu) else 0),
    transforms.ToTensor(),
    transforms.Normalize([0.45], [0.22])
])

slt_tfm = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomAffine(degrees=10),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomGrayscale(p=1 if (su or tu) else 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.45], [0.22])
])

fin_tfm = transforms.Compose([
    transforms.RandomChoice([slt_tfm, tfm])
])

ustm = transforms.Compose([
    transforms.Resize(img_size), 
    transforms.RandomOrder([
        transforms.RandomAffine(degrees=10),
        transforms.ColorJitter(0.2, 0.2),
    ]),
    transforms.ToTensor(),
    transforms.Normalize([0.25], [0.35])
])

uttm = transforms.Compose([
    transforms.Resize(img_size), 
    transforms.ToTensor(),
    transforms.Normalize([0.25], [0.35])
])

class CustomDataset(Dataset):
    def __init__(self, ds=None, mode=None, transform=None):
        # self.dm = join(args.offset, ds)
        self.datadir = args.offset
        # df = pd.read_csv(join(self.dm, '{}.csv'.format(mode)))

        self.filename = [f for f in listdir(self.datadir) if isfile(join(self.datadir, f))]
        # self.labels = df['label']
        # for i, name in enumerate(self.filename):
        #     self.filename[i] = name
        #     self.ids[i] = int(name.split('_')[-1].split('.')[0])
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(join(self.datadir, self.filename[index])).convert('RGB')
        # img = torchvision.io.read_image(fname)
        x = self.transform(img)
        return x, self.filename[index]

    def __len__(self):
        return len(self.filename)

# s_train = CustomDataset(mode='train', ds=args.sd, transform=fin_tfm if args.sd != 'usps' else ustm)
# s_test = CustomDataset(mode='test', ds=args.sd, transform=tfm if args.sd != 'usps' else uttm)
# t_train = CustomDataset(mode='train', ds=args.td, transform=fin_tfm if args.td != 'usps' else ustm)
t_test = CustomDataset(mode='test', ds=args.td, transform=tfm if args.td != 'usps' else uttm)
# mxlen, minlen = max(len(s_train), len(min))
# print(s_train.transform, s_test.transform, t_train.transform, t_test.transform)

'''s_train = ConcatDataset([s_train for i in range(len(t_train) // len(s_train))])\
             if len(t_train) > len(s_train) else s_train

t_train = ConcatDataset([t_train for i in range(len(s_train) // len(t_train))])\
             if len(s_train) > len(t_train) else t_train'''
# s_rld = DataLoader(s_train, shuffle=True, batch_size=batch_size, num_workers=2)
# t_rld = DataLoader(t_train, shuffle=True, batch_size=batch_size, num_workers=2)
# s_eld = DataLoader(s_test, shuffle=False, batch_size=batch_size)
t_eld = DataLoader(t_test, shuffle=False, batch_size=batch_size)

# print(len(s_train), len(t_train), len(s_rld), len(t_rld))

"""# Model

Feature Extractor: 典型的VGG-like疊法。

Label Predictor / Domain Classifier: MLP到尾。

相信作業寫到這邊大家對以下的Layer都很熟悉，因此不再贅述。
"""
from functions import ReverseLayerF    
class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

model = CNNModel().to(device)
model.load_state_dict(torch.load(join('p3', './{}.pt'.format(args.td)), map_location=device))

model.eval()
preds = []
truth = []
img_name = []
for i, (imgs, _) in enumerate((t_eld)):
    # print(imgs.shape, labels.shape)
    img_name.extend(_)
    imgs = imgs.to(device)
    # truth.extend(labels.numpy())
    # labels = labels.to(device)
    logits, _ = model(imgs, 0)
    # print(logits.shape)
    preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy())

# print(truth)
print(preds)
# acc = (np.sum(np.array(truth) == np.array(preds)) / len(preds))
with open(args.save_dir, "w") as f:

    # The first row must be "Id, Category"
    f.write("image_name,label\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(preds):
        f.write(f"{img_name[i]},{pred}\n")
# if acc > best_acc:
#     torch.save(model, join(args.model_dir, '{}_model_2.pt').format(args.sd))
# print('acc: {}'.format(acc))
# test(source_dataset_name, epoch)
# test(target_dataset_name, epoch)

print('done')
