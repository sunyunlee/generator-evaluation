import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from models.Classifier import Classifier
from data_functions.dataloaders import load_MNIST
from data_functions.dataprocessors import label_to_onehot
from plot_functions.plot_functions import plot_losses

SEED = 1234 

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


""" Dataset """
(X_train, Y_train), (X_test, Y_test) = load_MNIST("data")

""" Define variables """
N_CLASSES = 10 
INPUT_DIM = X_train.shape[1]
BATCH_SIZE = 64
N_EPOCHS = 1
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csf = Classifier(INPUT_DIM, N_CLASSES).type(torch.float64)

C_train = label_to_onehot(Y_train, N_CLASSES).type(torch.float64)
C_test = label_to_onehot(Y_test, N_CLASSES).type(torch.float64)

""" Iterators """
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE)

""" Loss function """ 
loss_fn = nn.NLLLoss()

""" Optimizer """
optimizer = torch.optim.Adam(csf.parameters(), lr=lr)

""" Train """
train_losses = []
test_losses = []

f = open("output/Classifier/csf_losses.txt", "a")
for ep in range(N_EPOCHS):
    for x, c in train_iter: 
        # Zero grad 
        optimizer.zero_grad() 
        
        # Forward 
        c_out = csf(x.to(device))
        
        # Loss 
        loss = loss_fn(c_out, c)
        
        # Backward 
        loss.backward()
        
        # Update 
        optimizer.step()
    
    with torch.no_grad(): 
        ## Train 
        # Forward 
        c_out = csf(X_train.to(device))
        
        # Loss 
        loss = loss_fn(c_out, Y_train)
        
        train_losses.append(loss.item())
        
        ## Test 
        # Forward 
        c_out = csf(X_test.to(device))
        
        # Loss 
        loss = loss_fn(c_out, Y_test)
        
        test_losses.append(loss.item())
        print("Epoch [%d / %d] train loss: %f test loss: %f" %(ep + 1, N_EPOCHS, train_losses[-1], test_losses[-1]))
        f.write("Epoch [%d / %d] train loss: %f test loss: %f" %(ep + 1, N_EPOCHS, train_losses[-1], test_losses[-1]))

f.close()

## Plot losses 
plot_losses("output/Classifier", "classifier", train_losses, test_losses)
