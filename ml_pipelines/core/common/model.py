from logging import Logger

import pandas as pd
from sklearn.linear_model import LogisticRegression


def predict(model: LogisticRegression, x_matrix: pd.DataFrame, logger: Logger):
    logger.info(f"Generating predictions for {len(x_matrix)} samples")
    return model.predict_proba(x_matrix)[:, 1]
