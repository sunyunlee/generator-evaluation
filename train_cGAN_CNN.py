from models.cGAN_CNN import Generator, Discriminator
from data_functions.dataloaders import load_MNIST
from data_functions.dataprocessors import scale_data, label_to_onehot
import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


SEED = 1234

OUTPUT_DIR = "output/cGAN-CNN"

if not os.path.exists(OUTPUT_DIR):
    print("Creating directory {}".format(OUTPUT_DIR))
    os.makedirs(OUTPUT_DIR)

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
N_EPOCHS = 100
N_CLASSES = 10
INPUT_H = 28
INPUT_W = 28
BATCH_SIZE = 100
LATENT_DIM = 50
GEN_LEARNING_RATE = 1e-3
GEN_STEPS_PER_CYCLE = 1
DISC_LEARNING_RATE = 1e-3
DISC_STEPS_PER_CYCLE = 1


def get_fake_labels(num):
    fake_classes = torch.Tensor([i % N_CLASSES for i in range(num)]).to(device)
    return label_to_onehot(fake_classes, N_CLASSES).float()


C_train = label_to_onehot(Y_train, N_CLASSES).float()
C_test = label_to_onehot(Y_test, N_CLASSES).float()

""" Declare generator and discriminator """
gen = Generator(LATENT_DIM, N_CLASSES, INPUT_H, INPUT_W)
disc = Discriminator(INPUT_H, INPUT_W, N_CLASSES)

""" Declare data iterators """
train_dataset = TensorDataset(X_train, C_train)
test_dataset = TensorDataset(X_test, C_test)

train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE)

""" Optimizers """
gen_opt = torch.optim.Adam(gen.parameters(), lr=GEN_LEARNING_RATE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=DISC_LEARNING_RATE)

""" Train """
gen_losses = []
disc_losses = []

zeroes = torch.Tensor(BATCH_SIZE).float().to(device).fill_(0.0)
ones = torch.Tensor(BATCH_SIZE).float().to(device).fill_(1.0)
fake_labels = get_fake_labels(BATCH_SIZE)

f = open(os.path.join(OUTPUT_DIR, "losses.txt"), "a")
for ep in range(N_EPOCHS):
    gen_loss = 0
    disc_loss = 0
    num_gen_steps = 0
    num_disc_steps = 0
    for real_images, real_labels in train_iter:
        # Train the discriminator
        for _ in range(DISC_STEPS_PER_CYCLE):
            # Zero grad 
            disc_opt.zero_grad()
            
            # Loss
            real_loss = F.mse_loss(disc(real_images, real_labels), ones)
            
            noise = sample_noise(BATCH_SIZE, LATENT_DIM)
            fake_images = gen(noise, fake_labels)
            fake_loss = F.mse_loss(disc(fake_images, fake_labels), zeroes)
            
            loss = 0.5 * real_loss + 0.5 * fake_loss
            
            # Backward
            loss.backward()

            disc_loss += loss.item()
            num_disc_steps += 1
            
            # Update 
            disc_opt.step()

        # Train the generator
        for _ in range(GEN_STEPS_PER_CYCLE):
            # Zero grad
            gen_opt.zero_grad()
            
            # Loss
            noise = sample_noise(BATCH_SIZE, LATENT_DIM)
            fake_images = gen(noise, fake_labels)
            loss = F.mse_loss(disc(fake_images, fake_labels), ones)

            gen_loss += loss.item()
            num_gen_steps += 1
            
            # Backward
            loss.backward()
            
            # Update
            gen_opt.step()
        
    gen_losses.append(gen_loss / num_gen_steps)
    disc_losses.append(disc_loss / num_disc_steps)
    to_write = "Epoch [%d / %d] gen. loss %.3f disc. loss %.3f"\
               % (ep + 1, N_EPOCHS, gen_losses[-1], disc_losses[-1])
    print(to_write)
    f.write(to_write + "\n")

f.close()

""" Save models """
torch.save(gen, os.path.join(OUTPUT_DIR, "generator.pt"))
torch.save(disc, os.path.join(OUTPUT_DIR, "discriminator.pt"))
