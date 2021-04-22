from typing import List
# from .MLP import MultilayerPerceptron
import torch.nn.functional as F
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, label_dim, latent_dim):
        super().__init__()
        # Implement CNN here 
        # 1 * 28 * 28 -> 16 * 12 * 12 -> 32 * 4 * 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.linear = nn.Linear(4*4*32 + label_dim, 300)
        self.mu_z = nn.Linear(300, latent_dim)
        self.logvar_z = nn.Linear(300, latent_dim)
    
    def forward(self, x, c):
        # Implement CNN here
        out = F.relu(self.conv1(x)) # [N, 16, 12, 12]
        out = F.relu(self.conv2(out)) # [N, 32, 4, 4]
        out = out.reshape((out.shape[0], -1)) # [N, 4*4*32]
        out = F.relu(self.linear(torch.cat([out, c], axis=1))) # [N, 300]
        mu_out = self.mu_z(out) # [N, latent_dim]
        logvar_out = self.logvar_z(out) # [N, latent_dim]
        
        return mu_out, logvar_out


class Decoder(nn.Module): 
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        # Implement CNN here 
        self.linear1 = nn.Linear(latent_dim + label_dim, 300)
        self.linear2 = nn.Linear(300,4*4*32)
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(1, 1, kernel_size=4)
    
    def forward(self, z, c):
        # Implement CNN here 
        out = F.relu(self.linear1(torch.cat([z, c], axis=1))) # [N, 300]
        out = F.relu(self.linear2(out)) # [N, 4*4*32]
        out = out.reshape((out.shape[0], 32, 4, 4)) # [N, 32, 4, 4]
        out = F.relu(self.conv1(out)) # [N, 16, 11, 11]
        out = F.relu(self.conv2(out)) # [N, 1, 25, 25]
        out = torch.sigmoid(self.conv3(out)) # [N, 1, 28, 28]
        return out

