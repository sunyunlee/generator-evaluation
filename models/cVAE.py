from typing import List
# from .MLP import MultilayerPerceptron
import torch.nn.functional as F
import torch
import torch.nn as nn


class Encoder(nn.Module): 
    def __init__(self, input_dim: int, label_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim + label_dim, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 100)
        self.mu_z = nn.Linear(100, latent_dim)
        self.log_var_z = nn.Linear(100, latent_dim)

    def forward(self, x, c):
        out = F.relu(self.linear1(torch.cat([x, c], axis=1)))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        mu_z = self.mu_z(out)
        log_var_z = self.log_var_z(out)
        
        return mu_z, log_var_z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, label_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super().__init__()
        
        self.linear1 = nn.Linear(latent_dim + label_dim, 100)
        self.linear2 = nn.Linear(100, 256)
        self.linear3 = nn.Linear(256, 512)
        self.output = nn.Linear(512, output_dim)

    def forward(self, z, c):
        out = F.relu(self.linear1(torch.cat([z, c], axis=1)))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = torch.sigmoid(self.output(out))
        
        return out
