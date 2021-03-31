from Model import Model 


class Generator(nn.Module): 
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        pass 
    
    def forward(self, z, c):
        pass


class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super().__init__()
        pass 

    def forward(self, x, c):
        pass

