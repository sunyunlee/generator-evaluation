from Model import Model 


class cVAE(Model):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()
        pass 

    def forward(self, x):
        # TODO: implement 
        pass

    def loss_function(self, x_pred, x_in):
        # TODO: implement
        pass

class Encoder(nn.Module): 
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()
        pass 

    def forward(self, x):
        # TODO: implement 
        pass

    def loss_function(self, x_pred, x_in):
        # TODO: implement
        pass


class Decoder(nn.Module):
    def __init__(self, latent, label_dim, hidden_dim, output_dim):
        super().__init__()
        pass 

    def forward(self, x):
        # TODO: implement 
        pass

    def loss_function(self, x_pred, x_in):
        # TODO: implement
        pass

