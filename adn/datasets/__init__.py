import os.path as path
from .deep_lesion import DeepLesion
from .spineweb import Spineweb

def get_dataset(dataset_type, **dataset_opts):
    return {
        "deep_lesion": DeepLesion,
        "spineweb": Spineweb
    }[dataset_type](**dataset_opts[dataset_type])