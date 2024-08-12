import sys

import mlflow
from omegaconf import DictConfig

from .pylogger import get_pylogger

log = get_pylogger(__name__)


def setup_mlflow(cfg: DictConfig) -> int:
    if cfg["task_name"] == "default":
        experiment_id = _setup_mlflow_local(cfg)
    elif cfg["task_name"] == "az_databricks":
        experiment_id = _setup_mlflow_azure_databricks(cfg)
    else:
        log.error(f"Unsupported task name : {cfg['task_name']}")
        sys.exit()
    return experiment_id


def _setup_mlflow_local(cfg: DictConfig):
    mlflow.set_tracking_uri(uri=cfg["logger"]["mlflow"]["tracking_uri"])
    # _experiment_name = cfg["logger"]["mlflow"]["experiment_name"] + f"({cfg['experience_name']})"
    _experiment_name = cfg["logger"]["mlflow"]["experiment_name"]
    try:
        experiment = mlflow.get_experiment_by_name(_experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(
            name=_experiment_name, artifact_location=cfg["logger"]["mlflow"]["artifact_location"]
        )

    log.info(f"Setup mlflow for local experiment")
    log.info(f"Experiment name : {_experiment_name}")
    log.info(f"Experiment_id : {experiment_id}")
    return experiment_id


def _setup_mlflow_azure_databricks(cfg: DictConfig):
    # _experiment_name = cfg["logger"]["mlflow"]["experiment_name"] + f"({cfg['experience_name']})"
    _experiment_name = cfg["logger"]["mlflow"]["experiment_name"]
    _experiment_name = f"/Users/{cfg['azure_user_name']}/{_experiment_name}"
    try:
        experiment = mlflow.get_experiment_by_name(_experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(name=_experiment_name)

    log.info(f"Setup mlflow for azure databricks experiment")
    log.info(f"Experiment name : {_experiment_name}")
    log.info(f"Experiment_id : {experiment_id}")
    return experiment_id
