import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import torch.nn.functional as F
import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x).convert('RGB')

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


def prototypical_loss(input, target, n_support, model, device):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    # print(query_idxs, target, support_idxs)
    query_samples = input.to('cpu')[query_idxs]
    # print(device)
    dists = model(query_samples.to(device), prototypes.to(device))
    # dists = euclidean_dist(query_samples.to(device), prototypes.to(device))
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    # print(log_p_y.shape, target_inds.shape)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long().to(device)

    # loss_val = criterion(dists, target_inds.squeeze(-1).view(-1))
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.squeeze(dim=-1).eq(target_inds.squeeze(dim=-1).squeeze(dim=-1)).float().mean()
    # print(y_hat, target_inds)
    return loss_val,  acc_val

def predict(args, model, mlp, data_loader):
    prediction_results = []
        # each batch represent one episode (support data + query data)
    for i, (data, target) in enumerate(data_loader):
        
        # split data into support and query data
        support_input = data[:args.N_way * args.N_shot,:,:,:].to(device) 
        query_input   = data[args.N_way * args.N_shot:,:,:,:].to(device)
        
        # create the relative label (0 ~ N_way-1) for query data
        label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
        query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])
        # support_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[:args.N_way * args.N_shot]])
        # print(support_label.shape)
        # print(label_encoder)
        # print(query_label)
        # TODO: extract the feature of support and query data
        mlp.load_state_dict(torch.load(os.path.join('p1', 'mlp0.pt'), map_location=device))
        classop = torch.optim.Adam(mlp.parameters(), lr=3e-5, weight_decay=1e-4)
        with torch.no_grad():
            z = model(support_input)
            z_proto = z[:args.N_way * args.N_shot].view(args.N_way, args.N_shot, z.shape[-1]).mean(1)
        
        z = z.detach()
        z_proto = z_proto.detach()
        # print(query_input.shape)
        mlp.train()
        # print(i)
        for e in range(12):
            zc = mlp(z, z_proto)
            log_p_y = F.log_softmax(-zc, dim=1).view(args.N_way, args.N_shot, -1)
            # print(log_p_y.shape, target_inds.shape)
            target_inds = torch.arange(0, args.N_way)
            target_inds = target_inds.view(args.N_way, 1, 1)
            target_inds = target_inds.expand(args.N_way, args.N_shot, 1).long().to(device)

            # loss_val = criterion(dists, target_inds.squeeze(-1).view(-1))
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            classop.zero_grad()
            loss_val.backward()
            classop.step()
            loss_val = loss_val.detach()
        
        zq = model(query_input)
        # TODO: calculate the prototype for each class according to its support data
        mlp.eval()
        with torch.no_grad():
            dists = mlp(zq, z_proto)
            target_inds = torch.arange(0, args.N_way)
            target_inds = target_inds.view(args.N_way, 1, 1)
            target_inds = target_inds.expand(args.N_way, args.N_shot, 1).long()
            log_p_y = F.log_softmax(-dists, dim=1)#.view(args.N_way, args.N_query, -1)
            # print(log_p_y.shape, dists.shape, zq.shape, z_proto.shape)
            _, y_hat = log_p_y.max(1)
            # print(log_p_y)
            # acc_val = torch.exq(y_hat.squeeze(), target_inds.squeeze()).float().mean()
            # TODO: classify the query data depending on the its distense with each prototype
            prediction_results.extend(y_hat.cpu().numpy().tolist())
    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N_way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N_shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N_query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")
    parser.add_argument('--offset', type=str, help="offset")
    parser.add_argument('--cuda', type=int, help="cuda", default=0)
    parser.add_argument('--ipc', type=int, help="ipc", default=600)
    parser.add_argument('--n_epoch', type=int, help="epochs", default=100)
    return parser.parse_args()


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, hid_dim=64, z_dim=64):
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)



class MLP(nn.Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear((2) * 1600, 800),
            # nn.BatchNorm1d(args.N_way),
            nn.ReLU(),
            nn.Linear(800, 400),
            # nn.BatchNorm1d(args.N_way),
            nn.ReLU(),
            nn.Linear(400, 200),
            # nn.BatchNorm1d(args.N_way),
            nn.ReLU(),
            nn.Linear(200, 1),
        )

    def forward(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        t = torch.cat([x, y], dim=-1)
        return self.mlp(t).squeeze(dim=-1)

def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


    


if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    device = 'cuda:{}'.format(args.cuda)
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(os.path.join('p1', 'model0.pt'), map_location=device))
    mlp = MLP().to(device)
    
    mlp.eval()
    model.eval()
    prediction_results = predict(args, model, mlp, test_loader)

    # TODO: output your prediction to csv

    df = pd.DataFrame(columns=['episode_id'] + ['query{}'.format(i) for i in range(args.N_way * args.N_query)])
    for b in range(len(prediction_results) // (args.N_way * args.N_query)):
        df.loc[len(df)] = [b] + prediction_results[b * (args.N_way * args.N_query): 
                                        (b + 1) * (args.N_way * args.N_query)]
    df.to_csv(args.output_csv, index=False)