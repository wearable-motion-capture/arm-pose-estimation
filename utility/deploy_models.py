from enum import Enum

from predict import models
from utility.names import NNS_INPUTS, NNS_TARGETS


class FF(Enum):
    H_XYZ = {
        "model": models.DropoutFF,
        "x_inputs": NNS_INPUTS.HACKATHON,
        "y_targets": NNS_TARGETS.H_XYZ,
        "hidden_layer_count": 4,
        "hidden_layer_size": 128,
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "dropout": 0.1,
        "sequence_len": 1,
        "normalize": False,
        "hash": "64ce7cfe3c6ba4918829617d8515f44772f4df04"
    }
