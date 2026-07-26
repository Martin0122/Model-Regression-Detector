import json

def load_dataset(path: str):
    dataset = {}
    with open(path, 'r') as f:
        dataset = json.load(f)
    return dataset

