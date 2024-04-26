import numpy as np
import torch

from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate import nn_models
from wear_mocap_ape.estimate.estimator import Estimator
from wear_mocap_ape.utility import transformations as ts
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class WatchPhoneUarmNN(Estimator):
    def __init__(self,
                 model_hash: str = deploy_models.LSTM.WATCH_PHONE_UARM.value,
                 smooth: int = 1,
                 add_mc_samples=True,
                 monte_carlo_samples=50,
                 bonemap: BoneMap = None,
                 tag: str = "NN UARM PHONE"):
        self.__tag = tag

        self._stream_mc = add_mc_samples
        self.__mc_samples = monte_carlo_samples

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

        # load model from given parameters
        self.__nn_model, params = nn_models.load_deployed_model_from_hash(hash_str=model_hash)

        super().__init__(
            x_inputs=NNS_INPUTS[params["x_inputs_n"]],
            y_targets=NNS_TARGETS[params["y_targets_n"]],
            smooth=smooth,
            normalize=params["normalize"],
            seq_len=params["sequence_len"],
            add_mc_samples=add_mc_samples,
            tag=tag,
            bonemap=bonemap
        )

    def parse_row_to_xx(self, row: np.array):
        if not type(row) is np.array:
            row = np.array(row)

        sw_fwd = np.array([
            row[self.__slp["sw_forward_w"]], row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]], row[self.__slp["sw_forward_z"]]
        ])
        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]], row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]], row[self.__slp["sw_rotvec_z"]]
        ])
        ph_fwd = np.array([
            row[self.__slp["ph_forward_w"]], row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]], row[self.__slp["ph_forward_z"]]
        ])

        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
        ])

        sw_sensor_dat = np.array([
            row[self.__slp["sw_dt"]],
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]]
        ])
        r_pres = row[self.__slp["sw_pres"]] - row[self.__slp["sw_init_pres"]]

        ph_sensor_dat = np.array([
            row[self.__slp["ph_gyro_x"]], row[self.__slp["ph_gyro_y"]], row[self.__slp["ph_gyro_z"]],
            row[self.__slp["ph_lvel_x"]], row[self.__slp["ph_lvel_y"]], row[self.__slp["ph_lvel_z"]],
            row[self.__slp["ph_lacc_x"]], row[self.__slp["ph_lacc_y"]], row[self.__slp["ph_lacc_z"]],
            row[self.__slp["ph_grav_x"]], row[self.__slp["ph_grav_y"]], row[self.__slp["ph_grav_z"]]
        ])

        # estimate north quat to align Z-axis of global and android coord system
        quat_north = ts.calib_watch_left_to_north_quat(sw_fwd)
        # the arm orientations if the calib position with left arm forward is perfect
        larm_dst_g = np.array([-0.7071068, 0, -0.7071068, 0])
        uarm_dst_g = np.array([0.7071068, 0, 0.7071068, 0])

        # calibrate watch with offset to perfect position
        sw_rot_g = ts.android_quat_to_global(sw_rot, quat_north)
        sw_fwd_g = ts.android_quat_to_global(sw_fwd, quat_north)
        sw_off_g = ts.hamilton_product(ts.quat_invert(sw_fwd_g), larm_dst_g)
        sw_cal_g = ts.hamilton_product(sw_rot_g, sw_off_g)

        # calibrate phone with offset to perfect position
        ph_rot_g = ts.android_quat_to_global(ph_rot, quat_north)
        ph_fwd_g = ts.android_quat_to_global(ph_fwd, quat_north)
        ph_off_g = ts.hamilton_product(ts.quat_invert(ph_fwd_g), uarm_dst_g)
        ph_cal_g = ts.hamilton_product(ph_rot_g, ph_off_g)

        return np.hstack([
            sw_sensor_dat,
            ts.quat_to_6drr_1x6(sw_cal_g),
            r_pres,
            ph_sensor_dat,
            ts.quat_to_6drr_1x6(ph_cal_g)
        ])

    def make_prediction_from_row_hist(self, xx):
        # cast to a torch tensor with batch size 1
        xx = torch.tensor(xx[None, :, :], dtype=torch.float32)
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