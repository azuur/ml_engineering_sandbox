import json
import os
import pickle
from pathlib import Path
from uuid import uuid4

import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from ml_pipelines.logic.common.feature_eng import FeatureEngineeringParams
from ml_pipelines.pipeline.train_pipeline import TrainArtifacts


def make_version(prefix: str) -> str:
    version = str(uuid4())
    return f"{prefix}_{version}"


def get_raw_data(version: str, root_path: os.PathLike) -> pd.DataFrame:
    return pd.read_csv(Path(root_path) / version / "raw_data.csv")


class ModelVersionAlreadyExists(Exception):
    pass


def save_train_artifacts(
    version: str,
    root_path: os.PathLike,
    artifacts: TrainArtifacts,
):
    version_dir = Path(root_path) / version
    try:
        version_dir.mkdir(parents=True)
    except FileExistsError:
        raise ModelVersionAlreadyExists()

    with open(version_dir / "model.pickle", "wb") as f:
        pickle.dump(artifacts["model"], f)
    with open(version_dir / "feature_eng_params.json", "w") as f:  # type: ignore
        f.write(artifacts["feature_eng_params"].json())
    artifacts["raw_train_data"].to_csv(version_dir / "raw_train_data.csv", index=False)
    artifacts["raw_test_data"].to_csv(version_dir / "raw_test_data.csv", index=False)
    artifacts["train_data"].to_csv(version_dir / "train_data.csv", index=False)
    artifacts["test_data"].to_csv(version_dir / "test_data.csv", index=False)


def get_train_artifacts(
    version: str,
    root_path: os.PathLike,
):
    version_dir = Path(root_path) / version
    with open(version_dir / "model.pickle", "rb") as f:  # type: ignore
        model: LogisticRegression = pickle.load(f)

    with open(version_dir / "feature_eng_params.json") as f:  # type: ignore
        feature_eng_params = FeatureEngineeringParams(**json.loads(f.read()))

    raw_train_data = pd.read_csv(version_dir / "raw_train_data.csv")
    raw_test_data = pd.read_csv(version_dir / "raw_test_data.csv")
    train_data = pd.read_csv(version_dir / "train_data.csv")
    test_data = pd.read_csv(version_dir / "test_data.csv")

    return TrainArtifacts(
        model=model,
        feature_eng_params=feature_eng_params,
        raw_train_data=raw_train_data,
        raw_test_data=raw_test_data,
        train_data=train_data,
        test_data=test_data,
    )


def save_eval_artifacts(
    version: str, root_path: os.PathLike, metrics: float, plots: Figure
):
    version_dir = Path(root_path) / version
    with open(version_dir / "metrics.txt", "w") as f:  # type: ignore
        f.write(str(metrics))  # type: ignore
    plots.savefig(str(version_dir / "calibration_plot.png"))


def get_latest_version(root_path: os.PathLike, filename: str) -> str:
    root_dir = Path(root_path)
    versions: list[tuple[str, float]] = []
    for version_dir in root_dir.iterdir():
        st_mtime = (version_dir / filename).stat().st_mtime
        versions.append((version_dir.stem, st_mtime))
    return max(versions, key=lambda t: t[1])[0]
