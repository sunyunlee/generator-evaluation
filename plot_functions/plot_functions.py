import os
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
# import seaborn as sns

COLORS = ["red", "pink", "orange", "yellow", "green", "teal", "blue", "purple", "brown", "gray"]


def plot_UMAP(generated_data: torch.tensor, labels: np.array, n_classes: int, model: str, path: str) -> None:
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(generated_data)
    labels = np.array(labels).flatten()

    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(n_classes):
        indices = np.where(labels==i)[0]
        plt.scatter(embedding[indices, 0], embedding[indices, 1], s=5, label=i, color=COLORS[i%10])
    plt.title("{} Generated MNIST Data".format(model), fontsize=18)
    plt.legend(markerscale=2)
    plt.savefig(os.path.join(path, '{}_UMAP.png'.format(model)))
