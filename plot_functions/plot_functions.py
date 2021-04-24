import os
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from typing import List
import torch 
# import seaborn as sns

COLORS = ["red", "pink", "orange", "yellow", "green", "teal", "blue", "purple", "brown", "gray"]


def plot_losses(path: str, model_name: str, train_losses: List[int], test_losses: List[int]) -> None:
    plt.plot(train_losses, "r")
    plt.plot(test_losses, "b")
    plt.ylabel("Losses")
    plt.xlabel("Epochs")
    plt.title("Train and Test Losses")
    plt.legend(["Train loss", "Valid Loss"])
    plt.savefig(os.path.join(path, "{}_losses.png".format(model_name)))


def plot_cgan_losses(path: str) -> None:
    gen_losses = []
    disc_losses = []
    gen_loss_prefix = "gen. loss "
    disc_loss_prefix = "disc. loss "
    loss_num_len = 5
    file = open(os.path.join(path, "losses.txt"), 'r')
    for line in file.readlines():
        if line.isspace():
            continue
        gen_loss_index = line.index(gen_loss_prefix) + len(gen_loss_prefix)
        gen_losses.append(float(line[gen_loss_index:(gen_loss_index + loss_num_len)]))
        disc_loss_index = line.index(disc_loss_prefix) + len(disc_loss_prefix)
        disc_losses.append(float(line[disc_loss_index:(disc_loss_index + loss_num_len)]))
    
    file.close()
    
    plt.plot(gen_losses, "r")
    plt.plot(disc_losses, "b")
    plt.ylabel("Losses")
    plt.xlabel("Epochs")
    plt.title("cGAN Component Losses")
    plt.legend(["Generator", "Discriminator"])
    plt.savefig(os.path.join(path, "losses.png"))


def plot_UMAP(generated_data: torch.tensor, labels: np.array, n_classes: int, model_name: str, path: str) -> None:
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(generated_data)
    labels = np.array(labels).flatten()

    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(n_classes):
        indices = np.where(labels==i)[0]
        plt.scatter(embedding[indices, 0], embedding[indices, 1], s=5, label=i, color=COLORS[i%10])
    plt.title("{} Generated MNIST Data".format(model_name), fontsize=18)
    plt.legend(markerscale=2)
    plt.savefig(os.path.join(path, '{}_UMAP.png'.format(model_name)))



