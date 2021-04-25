from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    
    def __init__(self, latent_dim, label_dim, output_h, output_w):
        super(Generator, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        
        self.linear1 = nn.Linear(latent_dim + label_dim, 300)
        self.linear2 = nn.Linear(300, 4*4*32)
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(1, 1, kernel_size=4)

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, labels), dim=1)
        out = F.leaky_relu(self.linear1(gen_input), 0.2) # [N, 300]
        out = F.leaky_relu(self.linear2(out), 0.2) # [N, 4*4*32]
        out = out.reshape((out.shape[0], 32, 4, 4)) # [N, 32, 4, 4]
        out = F.leaky_relu(self.conv1(out), 0.2) # [N, 16, 11, 11]
        out = F.leaky_relu(self.conv2(out), 0.2) # [N, 1, 25, 25]
        img = torch.sigmoid(self.conv3(out)) # [N, 1, 28, 28]
        return img.view(img.shape[0], self.output_h, self.output_w)


class Discriminator(nn.Module):
    
    def __init__(self, input_h, input_w, label_dim):
        super(Discriminator, self).__init__()
        # 1 * 28 * 28 -> 16 * 12 * 12 -> 32 * 4 * 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(4*4*32 + label_dim, 300)
        self.linear2 = nn.Linear(300, 1)

    def forward(self, img, labels):
        out = F.leaky_relu(self.conv1(img.view(img.shape[0], 1, 28, 28)), 0.2) # [N, 16, 12, 12]
        out = F.leaky_relu(self.conv2(out), 0.2) # [N, 32, 4, 4]
        out = out.reshape((out.shape[0], -1)) # [N, 4*4*32]
        out = F.leaky_relu(self.linear1(torch.cat((out.view(out.shape[0], -1), labels), dim=1)), 0.2) # [N, 300]
        disc_output = torch.sigmoid(self.linear2(out))
        return disc_output.view(disc_output.shape[0])

