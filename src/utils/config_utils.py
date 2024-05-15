from typing import Dict, Any

import os
import yaml

import utils.constants as constants


def load_model_config(
    name: str
) -> Dict[str, Any]:
    
    # get base config
    path = os.path.join(constants.MODEL_CONFIG_PATH, f"{name}.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_train_config(
    name: str,
) -> Dict[str, Any]:

    path = os.path.join(constants.TRAIN_CONFIG_PATH, f"{name}.yaml")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
