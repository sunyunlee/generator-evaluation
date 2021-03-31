# import torch.nn as nn 


# class MultilayerPerceptron(nn.Module):
#     def __init__(self, input_dim: int, hidden_dims: list, output_dim: int) -> None:
#         super().__init__()

#         self.input = nn.Linear(input_dim, hidden_dims[0])
#         if len(hidden_dims) > 1:
#             self.linears = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
#         else: 
#             self.linears = []
#         self.output = nn.Linear(hidden_dims[-1], output_dim)
    
#     def forward(self, x):
#         out = F.relu(self.input(x))
#         for i, l in enumerate(self.linears):
#             out = F.relu(l(out))
#         out = self.output(out)
        
#         return out
