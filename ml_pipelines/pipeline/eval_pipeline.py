import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_pipelines.logic.common.model import predict
from ml_pipelines.logic.eval.eval import (
    calculate_metrics,
    make_roc_plot,
    prob_calibration_plot,
)

# Input

with open("model.pickle", "rb") as f:
    model: LogisticRegression = pickle.load(f)

test_data = pd.read_csv("test_data.csv")


def eval_pipeline(model: LogisticRegression, data: pd.DataFrame):
    y_score = predict(model, data[["X1", "X2"]])
    metrics = calculate_metrics(data["Y"], y_score)
    roc_plot = make_roc_plot(model, data)
    calibration_plot = prob_calibration_plot(data, y_score)

    # Output
    plots, ax = plt.subplots(1, 2)
    roc_plot.plot(ax=ax[0])
    ax[0].plot([0, 1], [0, 1])
    calibration_plot(ax=ax[1])

    return (metrics, plots)


(metrics, plots) = eval_pipeline(model, test_data)

with open("metrics.txt", "w+") as f:  # type: ignore
    f.write(str(metrics))  # type: ignore
plots.savefig("calibration_plot.png")


# fig, ax = plt.subplots()
# for i in [0, 1]:
#     tmp = test_data.loc[test_data.Y == i, :]
#     ax.scatter(
#         tmp["X1"],
#         tmp["X2"],
#         c=tmp["Y"].map({0: "lightgray", 1: "red"}),
#         label=f"Y={i}",
#         s=2,
#         alpha=0.7,
#     )
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_title("Scatter plot of training data")
# ax.legend(framealpha=1)
# plt.show()
# plt.show()
