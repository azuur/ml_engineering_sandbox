import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score


def calculate_metrics(y_true: pd.Series, y_score: np.ndarray):
    return roc_auc_score(y_true, y_score)


def make_roc_plot(model: LogisticRegression, data: pd.DataFrame):
    display = RocCurveDisplay.from_estimator(model, data[["X1", "X2"]], data["Y"])
    return display


def prob_calibration_plot(data: pd.DataFrame, y_score: np.ndarray, step=0.005):
    xs = []
    ys = []

    m = np.max(y_score)
    m = min(1.1 * m, 0.3)

    for x in np.arange(0, 1, step).tolist():
        idx = np.logical_and(y_score >= x, y_score < x + step)
        y = data["Y"].loc[idx].mean()
        xs.append(x + step / 2)
        ys.append(y)

    X = np.vstack((np.ones(len(xs)), np.array(xs))).transpose()
    beta = 1.1 * np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(
        np.array(xs)
    )
    m_bins = beta[1] * m * 1.5

    def plot(ax):
        ax.scatter(xs, ys, c="red")
        ax.plot(xs, ys, c="red")
        ax.set_xlabel("Model scores")
        ax.set_ylabel("Observed frequency")
        ax.plot([0, m], [beta[0], beta[0] + beta[1] * m], c="red", alpha=0.5)
        ax.plot([0, m], [0, m])
        ax.set_xlim(-0.05, 1.1 * m_bins)
        ax.set_ylim(-0.05, 1.1 * m_bins)

    return plot
