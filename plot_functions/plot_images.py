import torch 
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_image_collage(path: str, model_name: str, images: torch.tensor, n_rows: int, n_cols: int) -> None:
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(100, 100)) # figsize(20, 20)

    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].imshow(images[row * n_rows + col], cmap="gray")
            axs[row][col].axis("off")

    plt.subplots_adjust(wspace=-0.3, hspace=0)
    plt.savefig(os.path.join(path, "{}_generated_images.png".format(model_name)))

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
