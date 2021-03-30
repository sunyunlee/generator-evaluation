import torch 
import numpy as np
import matplotlib.pyplot as plt


def plot_collage_MNIST(dataset, row: int, col: int):
    pass


def plot_single_MNIST(X: torch.tensor, y: int) -> None:
    """

    :param X: This is the MNIST image vector of size 784
    :param 
    """
    label = y
    img = X

    img = np.array(img, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    img = img.reshape((28, 28))

    # Plot
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(img, cmap='gray')
    plt.show()
