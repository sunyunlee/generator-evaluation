"""
Some code copied from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
"""
from typing import List
import torch
import torch.nn as nn


class Generator(nn.Module):
    
    def __init__(self, latent_dim, label_dim, hidden_dims: List[int],
                 output_h, output_w):
        super(Generator, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        
        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        hidden_block_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_block_layers.extend(block(hidden_dims[i], hidden_dims[i + 1]))
        
        self.model = nn.Sequential(
            *block(latent_dim + label_dim, hidden_dims[0], normalize=False),
            *hidden_block_layers,
            nn.Linear(hidden_dims[-1], output_h * output_w),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, labels), -1)
        img = self.model(gen_input)
        return img.view(img.shape[0], self.output_h, self.output_w)


class Discriminator(nn.Module):
    
    def __init__(self, input_h, input_w, label_dim, hidden_dims: List[int]):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, dropout=True):
            layers = [nn.Linear(in_features, out_features)]
            if dropout:
                layers.append(nn.Dropout(0.4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        hidden_block_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_block_layers.extend(block(hidden_dims[i], hidden_dims[i + 1]))
        
        self.model = nn.Sequential(
            *block(input_h * input_w + label_dim, hidden_dims[0], dropout=False),
            *hidden_block_layers,
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, img, labels):
        disc_input = torch.cat((img.view(img.shape[0], -1), labels), -1)
        disc_output = self.model(disc_input)
        return disc_output.view(disc_output.shape[0])
