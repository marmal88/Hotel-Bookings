import yaml

CONFIG_LOCATION = "./src/config/config.yaml"
CONFIG_TEST_LOCATION = "./tests/artifacts/test_config.yaml"


def read_yaml_file():
    """Load yaml file"""
    yaml_file = open(CONFIG_LOCATION, encoding="utf-8")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
    return config


def read_test_yaml_file():
    """Load test yaml file"""
    yaml_file = open(CONFIG_TEST_LOCATION, encoding="utf-8")
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
    return config
