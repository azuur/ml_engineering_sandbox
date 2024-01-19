import pandas as pd
from sklearn.linear_model import LogisticRegression


def predict(model: LogisticRegression, x_matrix: pd.DataFrame):
    return model.predict_proba(x_matrix)[:, 1]
