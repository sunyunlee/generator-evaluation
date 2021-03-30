import torch 
from typing import Tuple
from sklearn import preprocessing


def scale_data(train: torch.tensor, test: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    scaler = preprocessing.StandardScaler().fit(train)
    train = torch.tensor(scaler.transform(train)).type(torch.float64)
    test = torch.tensor(scaler.transform(test)).type(torch.float64)
    
    return train, test 


def label_to_onehot(labels: torch.tensors, n_classes: int) -> torch.tensor:
    N = labels.shape[0]
    onehot = torch.zeros(N, n_classes).type(torch.float64)
    onehot.scatter_(1, labels.type(torch.int64), torch.ones(N, 1).type(torch.float64))

    return onehot
