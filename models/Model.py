import torch.nn as nn 


class Model(nn.Module): 
    """ An abstract class for all model classes
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError

    def loss_function(self, x_pred, x_in):
        raise NotImplementedError

