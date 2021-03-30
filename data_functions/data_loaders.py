import torch
import torchvision
from torchvision import datasets, transforms
from typing import Tuple
from mlxtend.data import loadlocal_mnist

DATA_DIR = "../data"


def load_MNIST(path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """ Loads and returns MNIST train and test set

    :param scale: returns scaled dataset if True, defaults to False
    :return: train data, train label, test data, test label 
    """
    DATA_DIR = "./data"

    X_train, y_train = loadlocal_mnist(
        images_path='train-images-idx3-ubyte', 
        labels_path='train-labels-idx1-ubyte'
    )

    X_test, y_test = loadlocal_mnist(
        images_path="t10k-images-idx3-ubyte",
        labels_path="t10k-labels-idx1-ubyte"
    )

    X_train = torch.tensor(X_train / 255, dtype=torch.float64)
    Y_train = torch.tensor(y_train).reshape(X_train.shape[0], 1)
    X_test = torch.tensor(X_test / 255, dtype=torch.float64)
    Y_test = torch.tensor(y_test).reshape(X_test.shape[0], 1)

    return (X_train, y_train), (X_test, y_test)

    # train_dataset = datasets.MNIST(
    #     './data',
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose([transforms.ToTensor()]))

    # test_dataset = datasets.MNIST(
    #     './data',
    #     train=False,
    #     download=True,
    #     transform=transforms.Compose([transforms.ToTensor()])
    # )

    # return train_dataset, test_dataset
