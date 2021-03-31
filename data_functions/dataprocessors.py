import torch 
from typing import Tuple
from sklearn import preprocessing


def scale_data(train: torch.tensor, test: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    scaler = preprocessing.StandardScaler().fit(train)
    train = torch.tensor(scaler.transform(train)).type(torch.float64)
    test = torch.tensor(scaler.transform(test)).type(torch.float64)
    
    return train, test 


def label_to_onehot(labels: torch.tensor, n_classes: int) -> torch.tensor:

    onehot = torch.nn.functional.one_hot(labels.type(torch.int64).flatten()).type(torch.float64)

    return onehot
