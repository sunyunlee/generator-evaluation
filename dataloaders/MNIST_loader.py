import torchvision
from torchvision import datasets, transforms
from sklearn import preprocessing 
from typing import Tuple

    
def scale_data(train: torch.tensor, test: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    scaler = preprocessing.StandardScaler().fit(train)
    train = torch.tensor(scaler.transform(train)).type(torch.float64)
    test = torch.tensor(scaler.transform(test)).type(torch.float64)
    
    return train, test 


# def load_MNIST(scale: bool=False) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
def load_MNIST(scale: bool=False) -> Tuple[torch.tensor, torch.tensor]:
    """ Loads and returns MNIST train and test set

    :param scale: returns scaled dataset if True, defaults to False
    :return: train data, train label, test data, test label 
    """
    X_train = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor()
                                    ])),
        batch_size=batch_size_train, shuffle=True)
    
    # TODO: add label y_train when PyTorch MNIST issue is resolved 
    # https://github.com/pytorch/vision/issues/3497
    X_test = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor()
                                    ])),
        batch_size=batch_size_test, shuffle=True)
    # TODO: add label y_test
    if scale: 
        X_train, X_test = scale_data(X_train, X_test)
        
    # return X_train, y_train, X_test, y_test
    return X_train, X_test
