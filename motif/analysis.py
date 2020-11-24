import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.manifold import TSNE


def load_features(filename, scale=True):
    df = pd.read_csv(filename, index_col=0)
    X = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"].to_numpy()

    # normalize the data
    if scale:
        X = preprocessing.scale(X)

    return X, y


def plot_by_label(dim1, dim2, y):
    color_name = {
        0: "blue", 1: "orange", 2: "green", 3: "red", 4: "purple",
        5: "brown", 6: "pink", 7: "gray"
    }
    labels = [
        "Hip-Hop",
        "Pop",
        "Folk",
        "Experimental",
        "Rock",
        "International",
        "Electronic",
        "Instrumental",
    ]
    colors = ["tab:" + color_name[val] for val in y]

    plt.scatter(dim1, dim2, c=colors, alpha=1)
    for i, name in color_name.items():
        plt.scatter(
            [], [], c=("tab:" + name),
            marker=".", alpha=.3, label=labels[i])
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.legend()
    plt.show()


def scree_plot(filename="../data/features.csv"):
    X, y = load_features(filename)

    # compute the variance explained by each principal component
    pca = decomposition.PCA()
    pca.fit_transform(X)

    s = np.arange(1, X.shape[1] + 1)
    variance = pca.explained_variance_ratio_
    total_variance = sum(variance)

    plt.plot(s, [sum(variance[:i]) / total_variance for i in s], label="Cumulative")
    plt.plot(s, variance, label="Individual")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.suptitle("Variance Explained by Principal Components")
    plt.legend()
    plt.show()


def two_variable_pca(filename="../data/features.csv"):
    X, y = load_features(filename)

    # compute the first two principal components
    pc = decomposition.PCA(n_components=2).fit_transform(X)

    plot_by_label(pc[:, 0], pc[:, 1], y)


def t_sne(filename="../data/features.csv"):
    X, y = load_features(filename)

    # embed in two dimensions
    X_embedded = TSNE(n_components=2).fit_transform(X)

    plot_by_label(X_embedded[:, 0], X_embedded[:, 1], y)
