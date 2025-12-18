from yaml import safe_load

def get_config(city):
    config_path = "app/run.yml"
    with open(config_path) as file:
        config = safe_load(file)
    return config[city]
