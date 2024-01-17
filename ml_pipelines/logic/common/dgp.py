from functools import partial

import numpy as np
import pandas as pd


def p_y_given_x(x1, x2, theta: float = 0.0):
    alpha = np.cos(theta) - np.sin(theta)
    beta = np.sin(theta) + np.cos(theta)
    eta = -4 + 3 * (alpha * x1 + beta * x2) / (alpha + beta)
    return np.exp(eta) / (1 + np.exp(eta))


def generate_raw_data(n: int, seed: int, theta: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    data = pd.DataFrame({"X1": rng.random(n), "X2": rng.random(n)})
    train_probs = data.apply(
        lambda row: p_y_given_x(row["X1"], row["X2"], theta=theta), axis=1
    )
    data["Y"] = np.random.binomial(1, train_probs)
    data["X1"] = np.log(data["X1"] + 1)
    data["X2"] = np.exp(data["X2"])
    return data[["X1", "X2", "Y"]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    # Initialize an empty 2D array for Z
    # Z = np.zeros((len(x), len(y)))

    # # Evaluate the function for each combination of x and y
    # for i, y_v in enumerate(y):
    #     for j, x_v in enumerate(x):
    #         Z[i, j] = p_y_given_x(x_v, y_v, theta=-np.pi / 8)

    Z = partial(p_y_given_x, theta=np.pi / 8)(X, Y)

    # plt.pcolor(X, Y, Z)
    # plt.colorbar()
    # plt.show()
    data = generate_raw_data(5000, 813)
    data.to_csv("data.csv", index=False)
    plt.scatter(
        data["X1"],
        data["X2"],
        s=5,
        c=data["Y"].map({0: "lightgray", 1: "red"}),
        alpha=0.7,
    )
    plt.show()
