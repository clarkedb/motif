from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

genres = [
    "Hip-Hop",
    "Pop",
    "Folk",
    "Experimental",
    "Rock",
    "International",
    "Electronic",
    "Instrumental",
]


def plot_confusion_matrix(
    y_true, y_pred, labels, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.plot()
    plt.show()


def logistic_regression(
    filename="../data/features.csv", plot_matrix=False, test_size=0.1, normalize=False
):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf = linear_model.LogisticRegression().fit(x_train, y_train)

    predictions = clf.predict(x_test)
    print("Accuracy:", (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test))

    if plot_matrix:
        plot_confusion_matrix(y_test, predictions, genres, normalize=normalize)

    return clf
