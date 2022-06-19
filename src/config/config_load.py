import yaml

CONFIG_LOCATION = "./src/config/config.yaml"


def read_yaml_file():
    """Load yaml file"""
    yaml_file = open(CONFIG_LOCATION, encoding="utf-8")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
    return config
