import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            # Note: The original code used yaml.safe_load (implied by image_157a22.png)
            return yaml.safe_load(yaml_file)
    except Exception as e:
        # Raises custom exception with context
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)    