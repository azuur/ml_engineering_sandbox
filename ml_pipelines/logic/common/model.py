import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_model() -> LogisticRegression:
    with open("model.pickle", "rb") as f:
        return pickle.load(f)


def predict(model: LogisticRegression, x_matrix: pd.DataFrame):
    return model.predict_proba(x_matrix)[:, 1]
