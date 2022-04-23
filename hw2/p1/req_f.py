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
from typing import List



def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = computed_tmp - 1
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding

def filter2d(
        input: torch.Tensor, kernel: torch.Tensor, border_type: str = 'reflect', normalized: bool = False,
        padding: str = 'same'
    ) -> torch.Tensor:

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}")

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out

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

class Uvar:
    def __init__(self, device):
        self.batch_size = 1
        self.av = None
        # self.gan = GAN
        self.device = device
        same_seeds(2021)
    
    @torch.no_grad()
    def exists(self, val):
        return val is not None
    @torch.no_grad()
    def noise(self, n, latent_dim):
        return torch.randn(n, latent_dim).to(self.device)
    @torch.no_grad()
    def noise_list(self, n, layers, latent_dim):
        return [(self.noise(n, latent_dim), layers)]
    @torch.no_grad()
    def image_noise(self, n, im_size):
        return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(self.device)
    @torch.no_grad()
    def styles_def_to_tensor(self, styles_def):
        return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)
    @torch.no_grad()
    def evaluate_in_chunks(self, max_batch_size, model, *args):
        split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
        model.eval()
        # print(model.grad_fn)
        chunked_outputs = [model(*i) for i in split_args]
        if len(chunked_outputs) == 1:
            return chunked_outputs[0]
        return torch.cat(chunked_outputs, dim=0)
    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75, S=None, G=None):
        S = S
        
        latent_dim = G.latent_dim

        if not self.exists(self.av):
            z = self.noise(2000, latent_dim)
            samples = self.evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        
        av_torch = torch.from_numpy(self.av).to(self.device)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor
    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75, S=None, G=None):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi, S=S, G=G)            
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi, S=S, G=G)
        w_styles = self.styles_def_to_tensor(w_truncated)
        generated_images = self.evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)


    

    


