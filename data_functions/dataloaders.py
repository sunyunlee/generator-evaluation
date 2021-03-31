import torch
import torchvision
from torchvision import datasets, transforms
from typing import Tuple
from mlxtend.data import loadlocal_mnist
import os


def load_MNIST(path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """ Loads and returns MNIST train and test set

    :param scale: returns scaled dataset if True, defaults to False
    :return: train data, train label, test data, test label 
    """

    X_train, Y_train = loadlocal_mnist(
        images_path=os.path.join(path, 'train-images-idx3-ubyte'), 
        labels_path=os.path.join(path, 'train-labels-idx1-ubyte')
    )

    X_test, Y_test = loadlocal_mnist(
        images_path=os.path.join(path, "t10k-images-idx3-ubyte"),
        labels_path=os.path.join(path, "t10k-labels-idx1-ubyte")
    )

    X_train = torch.tensor(X_train / 255, dtype=torch.float64)
    Y_train = torch.tensor(Y_train).reshape(X_train.shape[0], 1)
    X_test = torch.tensor(X_test / 255, dtype=torch.float64)
    Y_test = torch.tensor(Y_test).reshape(X_test.shape[0], 1)

    return (X_train, Y_train), (X_test, Y_test)
