from models.cVAE import Encoder, Decoder 
from data_functions.dataloaders import load_MNIST
from data_functions.dataprocessors import scale_data, label_to_onehot
from plot_functions.plot_functions import plot_losses, plot_UMAP
from plot_functions.plot_images import plot_image_collage
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os


from torch.utils.data import DataLoader, TensorDataset
import numpy as np


SEED = 1234 

OUTPUT_DIR = "output/cVAE-MLP"

if not os.path.exists(OUTPUT_DIR):
    print("Creating directory {}".format(OUTPUT_DIR))
    os.makedirs(OUTPUT_DIR)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


(X_train, Y_train), (X_test, Y_test) = load_MNIST("data")


""" Define variables """
N_EPOCHS = 100
N_CLASSES = 10
INPUT_DIM = X_train.shape[1]
BATCH_SIZE = 64
LATENT_DIM = 50
lr = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

C_train = label_to_onehot(Y_train, N_CLASSES)
C_test = label_to_onehot(Y_test, N_CLASSES)

""" Declare encoder and decoder """
enc = Encoder(INPUT_DIM, N_CLASSES, [512, 256, 100], LATENT_DIM).type(torch.float64)
dec = Decoder(LATENT_DIM, N_CLASSES, [100, 256, 512], INPUT_DIM).type(torch.float64)

""" Declare data iterators """
train_dataset = TensorDataset(X_train, C_train)
test_dataset = TensorDataset(X_test, C_test)

train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE)

""" Optimizer """
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)

""" Loss function """
def loss_fn(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

""" Train """
train_losses = []
test_losses = []

f = open(os.path.join(OUTPUT_DIR, "cVAE_losses.txt"), "a")
for ep in range(N_EPOCHS):
    for x, c in train_iter: 
        # Zero grad 
        optimizer.zero_grad()
        
        # Forward 
        mu_z, log_var_z = enc(x.to(device), c.to(device))
        std_z = torch.exp(0.5 * log_var_z)
        
        eps = torch.randn_like(log_var_z)
        z_samples = eps.mul(std_z).add_(mu_z)
        x_out = dec(z_samples, c.to(device))
        
        # Loss 
        loss = loss_fn(x_out, x, mu_z, log_var_z)
        
        # Backward
        loss.backward()
        
        # Update 
        optimizer.step()
    
    with torch.no_grad():
        ## Train
        # Forward 
        mu_z, log_var_z = enc(X_train.to(device), C_train.to(device))
        std_z = torch.exp(0.5 * log_var_z)
        
        eps = torch.randn_like(log_var_z)
        z_samples = eps.mul(std_z).add_(mu_z)
        
        x_out = dec(z_samples, C_train.to(device))
        
        # Loss 
        loss = loss_fn(x_out, X_train, mu_z, log_var_z)
        train_losses.append(loss)
        
        ## Test 
        mu_z, log_var_z = enc(X_test.to(device), C_test.to(device))
        std_z = torch.exp(0.5 * log_var_z)
        
        eps = torch.randn_like(log_var_z)
        z_samples = eps.mul(std_z).add_(mu_z)
        
        x_out = dec(z_samples, C_test.to(device))
        
        # Loss 
        loss = loss_fn(x_out, X_test, mu_z, log_var_z)
        test_losses.append(loss)
        
        print("Epoch [%d / %d] train loss: %f test loss: %f" %(ep + 1, N_EPOCHS, train_losses[-1], test_losses[-1]))
        f.write("Epoch [%d / %d] train loss: %f test loss: %f \n" %(ep + 1, N_EPOCHS, train_losses[-1], test_losses[-1]))

f.close()

## Plot losses 
plot_losses(OUTPUT_DIR, "cVAE MLP", train_losses, test_losses)

""" Evaluate """ 
## MSE Loss ##
prior = torch.distributions.Normal(0, 1)
z = prior.sample((X_test.shape[0], LATENT_DIM)).type(torch.float64)
X_out = dec(z.to(device), C_test.to(device))
X_out = X_out.detach()
print("MSE Loss between generated data and test data: {}".format(torch.nn.MSELoss()(X_out, X_test).item()))

f = open(os.path.join(OUTPUT_DIR, "cVAE_MSELoss.txt"), "w")
f.write("MSE Loss between generated data and test data: {}".format(torch.nn.MSELoss()(X_out, X_test).item()))
f.close()

## Visualize generated images 
n_rows = N_CLASSES
n_cols = 10
N = n_rows * n_cols

prior = torch.distributions.Normal(0, 1)
z = prior.sample((N, LATENT_DIM)).type(torch.float64)
Y_gen = torch.tensor([i for i in range(n_cols) for j in range(n_rows)]).reshape(N, 1)
C_gen = label_to_onehot(Y_gen, N_CLASSES)

X_gen = dec(z.to(device), C_gen.to(device))
X_gen = X_gen.detach().numpy()

images = X_gen.reshape(N, 28, 28)
plot_image_collage(OUTPUT_DIR, "cVAE MLP", images, n_rows, n_cols)

## UMAP ##
import umap.umap_ as umap
import seaborn as sns

# Generate images 
N = 1000

prior = torch.distributions.Normal(0, 1)
z = prior.sample((N * N_CLASSES, LATENT_DIM)).type(torch.float64)

Y_gen = torch.tensor([i for i in range(N_CLASSES) for j in range(N)]).reshape(N * N_CLASSES, 1)
C_gen = label_to_onehot(Y_gen, N_CLASSES)

X_gen = dec(z.to(device), C_gen.to(device))
X_gen = X_gen.detach().numpy()

# Plot
plot_UMAP(X_gen, Y_gen, N_CLASSES, "cVAE MLP", OUTPUT_DIR)


## Save images for evaluation on classifier ##
N = X_test.shape[0]

prior = torch.distributions.Normal(0, 1)
z = prior.sample((N * N_CLASSES, LATENT_DIM)).type(torch.float64)

Y_gen = torch.tensor([i for i in range(N_CLASSES) for j in range(N)]).reshape(N * N_CLASSES, 1)
C_gen = label_to_onehot(Y_gen, N_CLASSES)

X_gen = dec(z.to(device), C_gen.to(device))
X_gen = X_gen.detach()

torch.save(X_gen, os.path.join(OUTPUT_DIR, "cVAE_MLP_generated_images.pt"))