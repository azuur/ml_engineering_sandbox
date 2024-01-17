import matplotlib.pyplot as plt
import pandas as pd

from ml_pipelines.logic.common.model import load_model, predict
from ml_pipelines.logic.eval.eval import (
    calculate_metrics,
    make_roc_plot,
    prob_calibration_plot,
)

# Input
model = load_model()
test_data = pd.read_csv("test_data.csv")

y_score = predict(model, test_data[["X1", "X2"]])
metrics = calculate_metrics(test_data["Y"], y_score)
roc_plot = make_roc_plot(model, test_data)
calibration_plot = prob_calibration_plot(test_data, y_score)

# Output
with open("metrics.txt", "w") as f:
    f.write(str(metrics))
fig, ax = plt.subplots()
roc_plot.plot(ax=ax)
ax.plot([0, 1], [0, 1])
plt.savefig("roc_plot.png")

fig, ax = plt.subplots()
calibration_plot(ax=ax)
plt.savefig("calibration_plot.png")

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
