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

    data_keys = [
        "raw_train_data",
        "raw_test_data",
        "train_data",
        "test_data",
    ]
    for key in data_keys:
        if key in artifacts:
            dataset = artifacts[key]  # type: ignore
            dataset.to_csv(version_dir / f"{key}.csv", index=False)


def get_train_artifacts(version: str, root_path: os.PathLike, load_data: bool = True):
    version_dir = Path(root_path) / version
    with open(version_dir / "model.pickle", "rb") as f:  # type: ignore
        model: LogisticRegression = pickle.load(f)

    with open(version_dir / "feature_eng_params.json") as f:  # type: ignore
        feature_eng_params = FeatureEngineeringParams(**json.loads(f.read()))

    if not load_data:
        return TrainArtifacts(
            model=model,
            feature_eng_params=feature_eng_params,
        )

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


def get_all_available_train_versions(root_path: os.PathLike | str):
    root_dir = Path(root_path)
    return [d.stem for d in root_dir.iterdir() if d.is_dir()]


def get_latest_version(root_path: os.PathLike, filename: str) -> str:
    root_dir = Path(root_path)
    versions: list[tuple[str, float]] = []
    for version_dir in root_dir.iterdir():
        if not version_dir.is_dir():
            continue
        st_mtime = (version_dir / filename).stat().st_mtime
        versions.append((version_dir.stem, st_mtime))
    return max(versions, key=lambda t: t[1])[0]


def get_best_version(train_artifacts_root_path: os.PathLike):
    train_dir = Path(train_artifacts_root_path)
    if "best_model" not in set(f for f in train_dir.iterdir() if f.is_file()):
        return None
    with open(train_dir / "best_model") as f:
        return f.read()


def tag_best_version(train_version: str, train_artifacts_root_path: os.PathLike):
    train_dir = Path(train_artifacts_root_path)
    with open(train_dir / "best_model", "w") as f:
        f.write(train_version)
