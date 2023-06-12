from enum import Enum

from predict import models
from utility.names import NNS_INPUTS, NNS_TARGETS


class FF(Enum):
    NORM_XYZ = {
        "model": models.DropoutFF,
        "x_inputs": NNS_INPUTS.MCSW,
        "y_targets": NNS_TARGETS.HAND_LARM_XYZ,
        "hidden_layer_count": 4,
        "hidden_layer_size": 128,
        "epochs": 200,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "dropout": 0.2,
        "sequence_len": 1
    }
    XYZ = {
        "model": models.DropoutFF,
        "x_inputs": NNS_INPUTS.MCSW,
        "y_targets": NNS_TARGETS.HAND_LARM_XYZ,
        "hidden_layer_count": 4,
        "hidden_layer_size": 128,
        "epochs": 200,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "dropout": 0.2,
        "sequence_len": 1,
        "normalize": False
    }
    NORM_UARM_LARM = {
        "model": models.DropoutFF,
        "x_inputs": NNS_INPUTS.MCSW,
        "y_targets": NNS_TARGETS.UARM_LARM_6DOF,
        "hidden_layer_count": 4,
        "hidden_layer_size": 128,
        "epochs": 200,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "dropout": 0.2,
        "sequence_len": 1,
        "normalize": True
    }
