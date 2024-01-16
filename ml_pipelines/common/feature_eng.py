import numpy as np
import pandas as pd


def transform_features(data: pd.DataFrame):
    data = data.copy()
    data["X1"] = np.exp(data["X1"]) - 1
    data["X2"] = np.log(data["X2"])
    return data
