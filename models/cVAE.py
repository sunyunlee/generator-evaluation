from typing import List
from Model import Model 
from MLP import MultilayerPerceptron


class Encoder(nn.Module): 
    def __init__(self, input_dim: int, label_dim: int, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__(input_dim + label_dim, hidden_dims, latent_dim)

    def forward(self, x):
        # TODO: implement 
        pass


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, label_dim: int, hidden_dims: List[int], output_dim: int) -> None:
        super().__init__(latent_dim + label_dim, hidden_dims, output_dim)

    def forward(self, x):
        # TODO: implement 
        pass
