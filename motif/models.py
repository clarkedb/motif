import ipyparallel as ipp
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import confusion_matrix
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

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
        filename="../data/features.csv",
        plot_matrix=False,
        test_size=0.3,
        normalize=False,
):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf = LogisticRegression().fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(
        "Logistic Regression Accuracy:",
        (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test),
    )

    if plot_matrix:
        plot_confusion_matrix(y_test, predictions, genres, normalize=normalize)

    return clf


def xgboost(
        filename="../data/features.csv", test_size=0.3, plot_matrix=False, normalize=True
):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    params = {
        "alpha": 1.3,
        "eta": 0.8,
        "gamma": 2.0,
        "lambda": 10.0,
        "num_class": 8,
        "objective": "multi:softmax",
    }

    # train model with best parameters from grid search
    bst = xgb.train(params, dtrain)

    predictions = bst.predict(dtest)
    print(
        "XGBoost Accuracy:",
        (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test),
    )

    if plot_matrix:
        plot_confusion_matrix(y_test, predictions, genres, normalize=normalize)

    return bst


def random_forest(
        filename="../data/features.csv", test_size=0.3, plot_matrix=False, normalize=True,
        print_feature_importance=False
):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=y)

    params = {"n_estimators": 1000}

    clf = RandomForestClassifier()
    clf.set_params(**params)
    clf.fit(x_train, y_train)

    if print_feature_importance:
        # get feature importance
        features = df.drop(["track_id", "genre_code"], axis=1).columns
        imp = clf.feature_importances_
        sorted_features = np.argsort(imp)

        print("Most-Important:", [features[i] for i in sorted_features[-3:]])
        print("Least-Important:", [features[i] for i in sorted_features[:3]])

    predictions = clf.predict(x_test)
    # print(
    #     "RF Accuracy:",
    #     (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test),
    # )

    if plot_matrix:
        plot_confusion_matrix(y_test, predictions, genres, normalize=normalize)

    return clf


def mlp(filename="../data/features.csv", test_size=0.3):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf = MLPClassifier(
        max_iter=500, warm_start=False, hidden_layer_sizes=700, alpha=0.002
    ).fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(
        "MLP Accuracy:",
        (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test),
    )


def knn(filename="../data/features.csv", test_size=0.3):
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    clf = KNeighborsClassifier(
        weights="distance", n_neighbors=100, leaf_size=50, algorithm="kd_tree"
    )
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    print(
        "KNN Accuracy:",
        (len(y_test) - np.count_nonzero(predictions - y_test)) / len(y_test),
    )


def tune_hyperparameters(filename="../data/features.csv", n_jobs=8, test_size=0.3):
    # load features
    df = pd.read_csv(filename, index_col=0)
    x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
    y = df["genre_code"]

    # setup parameter grid
    param_grid = {
        "C": np.arange(0.05, 1.2, 0.05),
        "l1_ratio": np.arange(0.005, 0.15, 0.005),
    }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # determine the score of one set of parameters
    def evaluate_params(g):
        clf = LogisticRegression(
            penalty="elasticnet", multi_class="multinomial", solver="saga"
        )
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


def xgboost_grid_search():
    """
    Run a grid search using an xgboost classifier and return the best set of parameters and its score
    """
    def evaluate_params(params):
        df = pd.read_csv("~/features.csv", index_col=0)
        x = preprocessing.scale(df.drop(["track_id", "genre_code"], axis=1))
        y = df["genre_code"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        n = len(y_test)

        booster = xgb.train(params, dtrain)
        predictions = booster.predict(dtest)
        accuracy = 1.0 - np.count_nonzero(predictions - y_test) / n
        return accuracy

    # setup parameter grid for grid search
    param_grid = list(ParameterGrid({
        "alpha": [0, .2, .5, .7, 1, 1.3, 2, 5, 10],  # default=0
        "eta": np.arange(0, 1.01, .1),  # default=0.3
        "gamma": [0, .5, .8, 1, 1.3, 2, 5, 20, 100, 500],  # default=0
        "lambda": [0, .2, .5, .7, 1, 1.3, 2, 5, 10],  # default=1
        "num_class": [8],
        "objective": ["multi:softmax"],
    }))

    # setup ipyparallel backend
    rc = ipp.Client()
    print(rc.ids)
    print(f"{len(rc.ids)} engines connected")
    print(f"{len(param_grid)} sets of parameters to test")
    lview = rc.load_balanced_view()

    # import necessary libraries on each engine
    lview.execute("import pandas as pd")
    lview.execute("from sklearn import preprocessing")
    lview.execute("import xgboost as xgb")

    # run a grid search
    scores = lview.map(evaluate_params, tqdm(param_grid))
    i = int(np.argmax(scores))

    return param_grid[i], scores[i]
