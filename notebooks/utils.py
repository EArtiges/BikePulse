from pathlib import Path

from yaml import safe_load


def get_config(city):
    # Get path relative to this file
    config_path = Path(__file__).parent / "run.yml"
    with open(config_path) as file:
        config = safe_load(file)
    return config[city]
