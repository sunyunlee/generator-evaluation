from data_functions.dataloaders import load_MNIST
from data_functions.dataprocessors import scale_data, label_to_onehot
from plot_functions.plot_functions import plot_cgan_losses, plot_UMAP
from plot_functions.plot_images import plot_image_collage
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


SEED = 1234


np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


(X_train, Y_train), (X_test, Y_test) = load_MNIST("data")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = X_train.float().to(device)
Y_train = Y_train.float().to(device)
X_test = X_test.float().to(device)
Y_test = Y_test.float().to(device)


def sample_noise(batch_size, dim):
    return (torch.rand(batch_size, dim) * 2 - 1).to(device)


""" Define variables """
N_CLASSES = 10
INPUT_H = 28
INPUT_W = 28
LATENT_DIM = 50


C_train = label_to_onehot(Y_train, N_CLASSES).float()
C_test = label_to_onehot(Y_test, N_CLASSES).float()


""" Load models """
OUTPUT_DIR = "output/cGAN-MLP"
# OUTPUT_DIR = "output/cGAN-CNN"

gen = torch.load(os.path.join(OUTPUT_DIR, "generator.pt"))
disc = torch.load(os.path.join(OUTPUT_DIR, "discriminator.pt"))


## Plot losses
plot_cgan_losses(OUTPUT_DIR)

""" Evaluate """
## MSE Loss ##
noise = sample_noise(X_test.shape[0], LATENT_DIM)
X_out = gen(noise, C_test)
X_out = X_out.view(X_out.shape[0], -1).detach()
loss = F.mse_loss(X_out, X_test).item()
print("MSE Loss between generated data and test data: {}".format(loss))

f = open(os.path.join(OUTPUT_DIR, "MSELoss.txt"), "w")
f.write("MSE Loss between generated data and test data: {}\n".format(loss))
f.close()

## Visualize generated images 
n_rows = N_CLASSES
n_cols = 10
N = n_rows * n_cols

noise = sample_noise(N, LATENT_DIM)
Y_gen = torch.tensor([(i % N_CLASSES) for i in range(N)]).reshape(N, 1)
C_gen = label_to_onehot(Y_gen, N_CLASSES).float()

X_gen = gen(noise, C_gen)
X_gen = X_gen.detach().numpy()

plot_image_collage(OUTPUT_DIR, "cGAN", X_gen.reshape(N, 28, 28), n_rows, n_cols)

## UMAP ##

# Generate images 
N = 1000

noise = sample_noise(N * N_CLASSES, LATENT_DIM)
Y_gen = torch.tensor([(i % N_CLASSES) for i in range(N * N_CLASSES)]).reshape(N * N_CLASSES, 1)
C_gen = label_to_onehot(Y_gen, N_CLASSES).float()

X_gen = gen(noise, C_gen)
X_gen = X_gen.view(X_gen.shape[0], -1).detach().numpy()

# Plot
plot_UMAP(X_gen, Y_gen, N_CLASSES, "cGAN", OUTPUT_DIR)


## Save images for evaluation on classifier ##
N = X_test.shape[0]

noise = sample_noise(N * N_CLASSES, LATENT_DIM)
Y_gen = torch.tensor([(i % N_CLASSES) for i in range(N * N_CLASSES)]).reshape(N * N_CLASSES, 1)
C_gen = label_to_onehot(Y_gen, N_CLASSES).float()

X_gen = gen(noise, C_gen)
X_gen = X_gen.view(X_gen.shape[0], -1).detach()

torch.save(X_gen, os.path.join(OUTPUT_DIR, 'generated_images.pt'))
