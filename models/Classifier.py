import torch 
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.linear = nn.Linear(4*4*32, 300)
        self.output = nn.Linear(300, output_dim)
        
    def forward(self, x):
        out = F.relu(self.conv1(x)) # [N, 16, 12, 12]
        out = F.relu(self.conv2(out)) # [N, 32, 4, 4]
        out = out.reshape((out.shape[0], -1)) # [N, 4*4*32]
        out = F.relu(self.linear(out)) # [N, 300]
        out = nn.Softmax(dim=1)(self.output(out)) # [N, latent_dim]

        return out



# class Classifier(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, 100)
#         self.bn1 = nn.BatchNorm1d(100)
#         self.dropout1 = nn.Dropout(0.5)
        
#         self.linear2 = nn.Linear(100, 50)
#         self.bn2 = nn.BatchNorm1d(50)
#         self.dropout2 = nn.Dropout(0.5)
        
#         self.output = nn.Linear(50, output_dim)
    
#     def forward(self, x):
#         out = self.linear1(x)
#         out = self.bn1(out)
#         out = F.relu(out)
#         out = self.dropout1(out)
        
#         out = self.linear2(out)
#         out = self.bn2(out)
#         out = F.relu(out)
#         out = self.dropout2(out)
        
#         out = self.output(out)
#         out = nn.Softmax(dim=1)(out)
#         return out 