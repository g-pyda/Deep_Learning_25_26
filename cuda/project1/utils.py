# helper functions (logging, metrics, etc.)
import yaml

def load_config(path="./config.yaml"):
    data = {}
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data