from typing import List
# from .MLP import MultilayerPerceptron
import torch.nn.functional as F
import torch
import torch.nn as nn


class Encoder(nn.Module): 
    def __init__(self, input_dim: int, label_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()

        self.input = nn.Linear(input_dim + label_dim, hidden_dims[0])
        if len(hidden_dims) > 1:
            self.linears = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        else: 
            self.linears = []
        self.mu_z = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_z = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x, c):
        out = F.relu(self.input(torch.cat([x, c], axis=1)))
        for i, l in enumerate(self.linears):
            out = F.relu(l(out))
        mu_z = self.mu_z(out)
        log_var_z = self.log_var_z(out)
        
        return mu_z, log_var_z


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, label_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super().__init__()

        self.input = nn.Linear(latent_dim + label_dim, hidden_dims[0])
        if len(hidden_dims) > 1:
            self.linears = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        else: 
            self.linears = []
        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z, c):
        out = F.relu(self.input(torch.cat([z, c], axis=1)))
        for i, l in enumerate(self.linears): 
            out = F.relu(l(out))
        out = torch.sigmoid(self.output(out))
        
        return out

