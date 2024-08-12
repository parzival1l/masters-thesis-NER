import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import tempfile
import pandas as pd
import hydra
import mlflow
import pyrootutils
from omegaconf import DictConfig
from tqdm import tqdm
from IngredientTaggingModel.train import Trainer
import util

os.environ["HYDRA_FULL_ERROR"] = "1"

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
config_dir = os.path.abspath("configs/")

log = util.get_pylogger(__name__)
@hydra.main(version_base=None, config_path=config_dir, config_name="az_databricks.yaml")
def main(cfg: DictConfig) -> None:

    experiment_id = util.setup_mlflow(cfg)
    log.info(f"Experiment id: {experiment_id}")

    with mlflow.start_run(
        run_name=cfg['experiment']['mlflow']['run_name'], experiment_id=experiment_id, description=cfg['experiment']['mlflow']["description"]
    ) as run:
        util.extras(cfg)
        exp_config = cfg["experiment"]
        trainer = Trainer(exp_config)
        trainer.validate()
        if cfg['task_name']=='az_databricks':
            mlflow.log_artifacts(mlflow.get_artifact_uri())
if __name__ == "__main__":
    main()
