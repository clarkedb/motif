from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import confusion_matrix
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

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
        filename="../data/features.csv", plot_matrix=False, test_size=0.3, normalize=False
):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf = LogisticRegression().fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print("Accuracy:", (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test))

    if plot_matrix:
        plot_confusion_matrix(y_test, predictions, genres, normalize=normalize)

    return clf


def tune_hyperparameters(filename="../data/features.csv", n_jobs=8, test_size=.3):
    # load features
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    # setup parameter grid
    param_grid = {
        "C": np.arange(.05, 1.2, .05),
        "l1_ratio": np.arange(.005, .15, .005)
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # determine the score of one set of parameters
    def evaluate_params(g):
        clf = LogisticRegression(penalty="elasticnet", multi_class="multinomial", solver="saga")
        clf.set_params(**g)
        clf.fit(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        return test_score, g

    #  evaluate each set of hyperparameters in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(g) for g in tqdm(ParameterGrid(param_grid))
    )

    # choose the best score
    best_score, best_grid, training_time = 0, {}, 0
    for score, g in results:
        if score > best_score:
            best_score = score
            best_grid = g

    return best_grid, best_score
