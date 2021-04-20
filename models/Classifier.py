import torch 
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.5)
        
        self.linear2 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.dropout2 = nn.Dropout(0.5)
        
        self.output = nn.Linear(50, output_dim)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.output(out)
        out = nn.Softmax(dim=1)(out)
        return out 