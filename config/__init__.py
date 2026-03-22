import yaml
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")

def load_config():
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config: dict):
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
