import enum
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
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy

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
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')
parser.add_argument("--save_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')
parser.add_argument("--model_dir", 
                    type=str, default='/tmp/b08902047/dlcv/hw2/hw2_data/digits/')
parser.add_argument("--sd", 
                    type=str, default='svhn')
parser.add_argument("--td", 
                    type=str, default='mnistm')

args = parser.parse_args()
args.model_dir = join(join(args.offset, args.td), 'ckpts')
device = 'cuda:' + args.cuda
batch_size = 64
print(args)

su = (args.sd == 'usps')
tu = (args.td == 'usps')

img_size = 28 if (su or tu) else 32


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



s_tfm = transforms.Compose([
    transforms.RandomChoice([ImageNetPolicy(), CIFAR10Policy(), SVHNPolicy()]), 
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

urm = transforms.Compose([
    s_tfm,
    transforms.Normalize([0.4], [0.3])
])

t_tfm = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.4], [0.3])
])

class CustomDataset(Dataset):
    def __init__(self, ds=None, mode=None, transform=None):
        self.dm = join(args.offset, ds)
        self.datadir = join(self.dm, mode)
        df = pd.read_csv(join(self.dm, '{}.csv'.format(mode)))

        self.filename = [e for e in df['image_name']]
        self.labels = df['label']
        # for i, name in enumerate(self.filename):
        #     self.filename[i] = name
        #     self.ids[i] = int(name.split('_')[-1].split('.')[0])
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(join(self.datadir, self.filename[index])).convert('RGB')
        # img = torchvision.io.read_image(fname)
        x = self.transform(img)
        return x, self.labels[index]

    def __len__(self):
        return len(self.filename)

# s_train = CustomDataset(mode='train', ds=args.sd, transform=fin_tfm if not su else ustm)
# s_test = CustomDataset(mode='test', ds=args.sd, transform=tfm if not su else uttm)
# t_train = CustomDataset(mode='train', ds=args.td, transform=fin_tfm if not tu else ustm)
# t_test = CustomDataset(mode='test', ds=args.td, transform=tfm if not tu else uttm)
s_train = CustomDataset(mode='train', ds=args.sd, transform=urm)
s_test = CustomDataset(mode='test', ds=args.sd, transform=t_tfm)
t_train = CustomDataset(mode='train', ds=args.td, transform=urm)
t_test = CustomDataset(mode='test', ds=args.td, transform=t_tfm)

# mxlen, minlen = max(len(s_train), len(min))
print(s_train.transform, s_test.transform, t_train.transform, t_test.transform)

s_train = ConcatDataset([s_train for i in range(len(t_train) // len(s_train))])\
             if len(t_train) > len(s_train) else s_train

t_train = ConcatDataset([t_train for i in range(len(s_train) // len(t_train))])\
             if len(s_train) > len(t_train) else t_train

s_rld = DataLoader(s_train, shuffle=True, batch_size=batch_size, num_workers=2)
t_rld = DataLoader(t_train, shuffle=True, batch_size=batch_size, num_workers=2)
s_eld = DataLoader(s_test, shuffle=False, batch_size=batch_size)
t_eld = DataLoader(t_test, shuffle=False, batch_size=batch_size)

# print(len(s_train), len(t_train), len(s_rld), len(t_rld))

"""# Model

Feature Extractor: 典型的VGG-like疊法。

Label Predictor / Domain Classifier: MLP到尾。

相信作業寫到這邊大家對以下的Layer都很熟悉，因此不再贅述。
"""

from models.build_gen import *


G = Generator(args.sd, args.td).to(device)
C1 = Classifier(source=args.sd, target=args.td).to(device)
C2 = Classifier(source=args.sd, target=args.td).to(device)
opt_g = optim.Adam(G.parameters(),
                    lr=args.lr, weight_decay=0.0005)
opt_c1 = optim.Adam(C1.parameters(),
                    lr=args.lr, weight_decay=0.0005)
opt_c2 = optim.Adam(C2.parameters(),
                    lr=args.lr, weight_decay=0.0005)

def reset_grad():
    opt_g.zero_grad()
    opt_c1.zero_grad()
    opt_c2.zero_grad()

def ent(output):
        return - torch.mean(output * torch.log(output + 1e-6))


def discrepancy(out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))

from tqdm.auto import tqdm
n_epoch = args.n_epoch
# loss_class = nn.CrossEntropyLoss()
# loss_domain = nn.BCEWithLogitsLoss()
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()
criterion = nn.CrossEntropyLoss().to(device)

# all_s = torch.cat([e[0].unsqueeze(0) for e in s_train], dim=0)
# all_t = torch.cat([e[0].unsqueeze(0) for e in t_train], dim=0)

# print(torch.mean(all_s, dim=[0, 2, 3]), torch.std(all_s, dim=[0, 2, 3]))
# print(torch.mean(all_t, dim=[0, 2, 3]), torch.std(all_t, dim=[0, 2, 3]))

best_acc = 0
num_k = 4
for epoch in (range(n_epoch)):

    len_dataloader = min(len(s_rld), len(t_rld))
    data_source_iter = iter(s_rld)
    data_target_iter = iter(t_rld)
    G.train()
    C1.train()
    C2.train()
    l1 = []
    l2 = []
    l3 = []
    for i in tqdm(range(len_dataloader)):
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(s_label)
        img_s = s_img.to(device)
        img_t = t_img.to(device)
        '''imgs = Variable(torch.cat((img_s, \
                                    img_t), 0))'''
        label_s = Variable(s_label.long().to(device))

        img_s = Variable(img_s)
        img_t = Variable(img_t)
        reset_grad()
        feat_s = G(img_s)
        output_s1 = C1(feat_s)
        output_s2 = C2(feat_s)

        loss_s1 = criterion(output_s1, label_s)
        loss_s2 = criterion(output_s2, label_s)
        loss_s = loss_s1 + loss_s2
        loss_s.backward()
        opt_g.step()
        opt_c1.step()
        opt_c2.step()
        reset_grad()

        feat_s = G(img_s)
        output_s1 = C1(feat_s)
        output_s2 = C2(feat_s)
        feat_t = G(img_t)
        output_t1 = C1(feat_t)
        output_t2 = C2(feat_t)

        loss_s1 = criterion(output_s1, label_s)
        loss_s2 = criterion(output_s2, label_s)
        loss_s = loss_s1 + loss_s2
        loss_dis = discrepancy(output_t1, output_t2)
        loss = loss_s - loss_dis
        loss.backward()
        opt_c1.step()
        opt_c2.step()
        reset_grad()
        l1.append(loss_s1.item())
        l2.append(loss_s2.item())
        for i in range(num_k):
            #
            feat_t = G(img_t)
            output_t1 = C1(feat_t)
            output_t2 = C2(feat_t)
            loss_dis = discrepancy(output_t1, output_t2)
            loss_dis.backward()
            opt_g.step()
            reset_grad()
            l3.append(loss_dis.item())
        '''if batch_idx > 500:
            return batch_idx

        if batch_idx % interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                epoch, batch_idx, 100,
                100. * batch_idx / 70000, loss_s1.data[0], loss_s2.data[0], loss_dis.data[0]))
            if record_file:
                record = open(record_file, 'a')
                record.write('%s %s %s\n' % (loss_dis.data[0], loss_s1.data[0], loss_s2.data[0]))
                record.close()
    return batch_idx'''

    print('epoch: {:03d} / {:03d}, loss1: {}, loss2: {}, disc: {}'.
            format(epoch + 1, n_epoch, np.mean(l1), np.mean(l2), np.mean(l3)))
            
        
    G.eval()
    C1.eval()
    C2.eval()
    preds = []
    truth = []
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    size = 0
    for i, (imgs, labels) in enumerate(tqdm(t_eld)):
        # print(imgs.shape, labels.shape)
        imgs = imgs.to(device)
        truth.extend(labels.numpy())
        labels = labels.to(device)
        feat = G(imgs)
        output1 = C1(feat)
        output2 = C2(feat)
        # print(output1.shape)
        test_loss += F.nll_loss(output1, labels).item()
        output_ensemble = output1 + output2
        pred1 = output1.data.max(1)[1]
        pred2 = output2.data.max(1)[1]
        pred_ensemble = output_ensemble.data.max(1)[1]
        preds.extend(torch.argmax(output_ensemble, dim=-1).detach().cpu().numpy())
        k = labels.data.size()[0]
        correct1 += pred1.eq(labels.data).cpu().sum()
        correct2 += pred2.eq(labels.data).cpu().sum()
        correct3 += pred_ensemble.eq(labels.data).cpu().sum()
        # logits, _ = model(imgs, alpha)
        # print(logits.shape)
        # preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy())

    # print(truth)
    # print(preds)
    acc = (np.sum(np.array(truth) == np.array(preds)) / len(preds))
    # acc = 0
    if acc > best_acc:
        torch.save(G.state_dict(), join(args.model_dir, '{}_Gmcd_v2.pt').format(args.td))
        torch.save(C1.state_dict(), join(args.model_dir, '{}_C1mcd_v2.pt').format(args.td))
        torch.save(C2.state_dict(), join(args.model_dir, '{}_C2mcd_v2.pt').format(args.td))
        best_acc = acc
    print('epoch: {:03d} / {:03d}, acc: {}, loss: {:04f}'.format(epoch + 1, n_epoch, acc, test_loss))
    # test(source_dataset_name, epoch)
    # test(target_dataset_name, epoch)

print('done')
