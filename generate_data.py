from data_functions.dataloaders import load_MNIST
from data_functions.dataprocessors import label_to_onehot
from plot_functions.plot_images import plot_image_collage

import torch 
import numpy as np
import os
import pandas as pd


TRAINED_CSF_PATH = "output/Classifier/classifier.pt"

# TODO: constant to set
MODEL = "cVAE-MLP"
# MODEL = "cVAE-CNN" 
# MODEL = "cGAN-MLP"
# MODEL = "cGAN-CNN"

LATENT_DIM = 50 # TODO: constant to set

SEED = 1234

N_CLASSES = 10
OUTPUT_DIR = "classifier-results/{}".format(MODEL)
TRAINED_MODEL_PATH = "output/{}/decoder.pt".format(MODEL)

device = torch.device("cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if not os.path.exists(OUTPUT_DIR):
    print("Creating directory {}".format(OUTPUT_DIR))
    os.makedirs(OUTPUT_DIR)

# I. Get data

(X_train, Y_train), (X_test, Y_test) = load_MNIST("data")

C_train = label_to_onehot(Y_train, N_CLASSES)
C_test = label_to_onehot(Y_test, N_CLASSES)

# II. Get trained model 
dec = torch.load(TRAINED_MODEL_PATH)
csf = torch.load(TRAINED_CSF_PATH)

# III. Get outputs from trained model 
# # Test trained model 
# n_rows = N_CLASSES
# n_cols = 10
# N = n_rows * n_cols

# prior = torch.distributions.Normal(0, 1)
# z = prior.sample((N, LATENT_DIM)).type(torch.float64)
# Y_gen = torch.tensor([i for i in range(n_cols) for j in range(n_rows)]).reshape(N, 1)
# C_gen = label_to_onehot(Y_gen, N_CLASSES)
# X_gen = dec(z.to(device), C_gen.to(device))
# X_gen = X_gen.detach().numpy()

# images = X_gen.reshape(N, 28, 28)
# plot_image_collage(OUTPUT_DIR, "cVAE-MLP", images, n_rows, n_cols)

# Get outputs from model 
N_PER_CLASS = 1000
N = N_CLASSES * N_PER_CLASS

prior = torch.distributions.Normal(0, 1)
z = prior.sample((N, LATENT_DIM)).type(torch.float64)
Y_gen = torch.tensor([i for i in range(N_CLASSES) for j in range(N_PER_CLASS)]).reshape(N, 1)

indices = torch.randperm(N)
Y_gen = Y_gen[indices]
C_gen = label_to_onehot(Y_gen, N_CLASSES)
X_gen = dec(z.to(device), C_gen.to(device))
X_gen = X_gen.reshape(len(X_gen), 1, 28, 28)


# IV. Pass outputs from the trained model to classifier
c_out = csf(X_gen)
class_out = torch.argmax(c_out, 1)
count_correct = torch.sum(class_out.flatten() == Y_gen.flatten())
accuracy = count_correct / len(Y_gen)

print("The accuracy score: ", accuracy.item())

f = open(os.path.join(OUTPUT_DIR, "{}_losses.txt".format(MODEL)), "a")
f.write("The accuracy score: {}".format(accuracy.item()))
f.close()


#PCA 
LABEL_DIM = 10
x = X_gen.reshape(N, 28*28).detach().numpy()

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.mixture import GaussianMixture

x_pca = PCA(n_components=2).fit_transform(x)
df_pca = pd.DataFrame(x_pca, columns=["principal component 1", "principal component 2"])
df_pca_labels = pd.concat([df_pca, pd.DataFrame(np.array(Y_gen), columns=["labels"])], axis=1)

gmm_pred_labels = GaussianMixture(n_components=LABEL_DIM, reg_covar=1e-5).fit_predict(x)

df_pca_gmm_labels = pd.concat([df_pca_labels, pd.DataFrame(gmm_pred_labels, columns=["gmm_labels"])], axis=1)

from collections import Counter

asgnd_gmm_labels = np.unique(np.array(df_pca_gmm_labels["gmm_labels"])).astype(int)
corr_gmm_labels = []

for i in asgnd_gmm_labels: 
    most_common = Counter(df_pca_gmm_labels[df_pca_gmm_labels["gmm_labels"] == i]["labels"]).most_common()[0][0]
    corr_gmm_labels.append(most_common)

df_pca_gmm_labels["corr_gmm_labels"] = np.array(corr_gmm_labels)[df_pca_gmm_labels["gmm_labels"]]

labels_true = list(df_pca_gmm_labels["labels"])
labels_pred = list(df_pca_gmm_labels["corr_gmm_labels"])
print("Gaussian Mixture v measure: ", metrics.v_measure_score(labels_true, labels_pred))

from sklearn.cluster import KMeans

kmeans_pred_labels = KMeans(n_clusters=LABEL_DIM, init="k-means++").fit_predict(x)
df_pca_kmean_labels = pd.concat([df_pca_labels, pd.DataFrame(kmeans_pred_labels, columns=["km_labels"])], axis=1)

asgnd_km_labels = np.unique(np.array(df_pca_kmean_labels["km_labels"])).astype(int)
corr_km_labels = []

for i in asgnd_km_labels: 
    most_common = Counter(df_pca_kmean_labels[df_pca_kmean_labels["km_labels"] == i]["labels"]).most_common()[0][0]
    corr_km_labels.append(most_common)

df_pca_kmean_labels["corr_km_labels"] = np.array(corr_km_labels)[df_pca_kmean_labels["km_labels"]]

labels_true = list(df_pca_kmean_labels["labels"])
labels_pred = list(df_pca_kmean_labels["corr_km_labels"])
print("KMeans v measure: ", metrics.v_measure_score(labels_true, labels_pred))
