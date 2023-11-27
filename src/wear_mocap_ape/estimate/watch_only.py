from abc import abstractmethod

import torch
import numpy as np

from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.estimate import models
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate.estimator import Estimator
from wear_mocap_ape.utility import transformations as ts
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class WatchOnly(Estimator):
    def __init__(self,
                 model_hash: str = deploy_models.LSTM.WATCH_ONLY.value,
                 smooth: int = 10,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25,
                 bonemap: BoneMap = None,
                 tag: str = "PUB WATCH"):
        self.__tag = tag

        # monte carlo predictions
        self.__stream_mc = stream_monte_carlo
        self.__mc_samples = monte_carlo_samples

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_ONLY_IMU_LOOKUP

        # load the trained network
        torch.set_default_dtype(torch.float64)

        # load model from given parameters
        self.__nn_model, params = models.load_deployed_model_from_hash(hash_str=model_hash)
        super().__init__(
            x_inputs=NNS_INPUTS(params["x_inputs_v"]),
            y_targets=NNS_TARGETS(params["y_targets_v"]),
            smooth=smooth,
            normalize=params["normalize"],
            seq_len=params["sequence_len"],
            stream_mc=self.__stream_mc,
            tag=tag,
            bonemap=bonemap
        )

    def parse_row_to_xx(self, row) -> np.array:
        # process the data
        # pressure - calibrated initial pressure = relative pressure
        r_pres = row[self.__slp["sw_pres"]] - row[self.__slp["sw_init_pres"]]

        # calibrate smartwatch rotation
        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]],
            row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]],
            row[self.__slp["sw_rotvec_z"]]
        ])
        sw_fwd = np.array([
            row[self.__slp["sw_forward_w"]],
            row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]],
            row[self.__slp["sw_forward_z"]]
        ])
        quat_north = ts.calib_watch_left_to_north_quat(sw_fwd)
        sw_quat_cal = ts.android_quat_to_global(sw_rot, quat_north)
        sw_6drr_cal = ts.quat_to_6drr_1x6(sw_quat_cal)  # get 6dof rotation representation

        # assemble the entire input vector of one time step
        return np.hstack([
            row[self.__slp["sw_dt"]],
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]],
            sw_6drr_cal,
            r_pres
        ])

    def make_prediction_from_row_hist(self, xx_hist: np.array) -> np.array:
        # cast to a torch tensor with batch size 1
        xx = torch.tensor(xx_hist[None, :, :])
        with torch.no_grad():
            # make mote carlo predictions if the model makes use of dropout
            t_preds = self.__nn_model.monte_carlo_predictions(x=xx, n_samples=self.__mc_samples)

        # if on GPU copy the tensor to host memory first
        if self._device.type == 'cuda':
            t_preds = t_preds.cpu()
        t_preds = t_preds.numpy()

        # we are only interested in the last prediction of the sequence
        t_preds = t_preds[:, -1, :]
        return t_preds

    @abstractmethod
    def process_msg(self, msg: np.array):
        return
