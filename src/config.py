import yaml
import os

from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, yaml_path="config.yaml"):
        with open(yaml_path, "r") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, item):
        return self.data[item]
