from Model import Model 


class cGAN(Model): 
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()
        pass 

    def forward(self, x):
        # TODO: implement 
        # Use Generator and Discriminator 
        pass

    def loss_function(self, x_pred, x_in):
        # TODO: implement
        pass


class Generator(nn.Module): 
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        pass 

    def forward(self, x):
        # TODO: implement 
        pass


class Discriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super().__init__()
        pass 

    def forward(self, x):
        # TODO: implement 
        pass
