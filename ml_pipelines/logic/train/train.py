from logging import Logger

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, random_state: int, logger: Logger):
    logger.info(f"Splitting data for train/test. {random_state = }")
    train_data, test_data = train_test_split(
        data,
        train_size=0.8,
        random_state=random_state,
        stratify=data["Y"],
        shuffle=True,
    )
    return train_data, test_data


def train_model(train_data: pd.DataFrame, logger: Logger):
    logger.info("Training model.")
    x_matrix = train_data[["X1", "X2"]]
    y_matrix = train_data["Y"]

    model = LogisticRegression(penalty=None)
    model.fit(X=x_matrix, y=y_matrix)
    logger.info("Model trained.")
    return model
