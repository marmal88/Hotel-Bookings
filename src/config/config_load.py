import yaml

config_location = "./src/config/config.yaml"


def read_yaml_file():
    """Load yaml file"""
    yaml_file = open(config_location)
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config
