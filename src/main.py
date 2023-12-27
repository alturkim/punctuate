import os
import hydra
from omegaconf import DictConfig
from prepare_dataset import download_dataset, clean_dataset, preprocess_dataset

_steps = [
    # "download",
    # "clean",
    "preprocess"
]

# This automatically reads in the configuration
@hydra.main(config_name='config', version_base=None, config_path='.')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = _steps

    root_path = hydra.utils.get_original_cwd()
    data_root = os.path.join(root_path, "data")

    if "download" in active_steps:
        download_dataset.go(config)

    if "clean" in active_steps:
        clean_dataset.go(config)

    if "preprocess" in active_steps:
        preprocess_dataset.go(config)

if __name__ == "__main__":
    go()

