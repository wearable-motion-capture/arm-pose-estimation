from abc import abstractmethod
from einops import rearrange

import numpy as np
import torch

from wear_mocap_ape.estimate import kalman_models
from wear_mocap_ape import config
from wear_mocap_ape.estimate.estimator import Estimator
from wear_mocap_ape.utility import transformations as ts
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class WatchPhonePocketKalman(Estimator):
    def __init__(self,
                 smooth: int = 1,
                 num_ensemble: int = 32,
                 model_hash: str = "SW-model-sept-4",
                 window_size: int = 10,
                 stream_mc: bool = True,
                 tag: str = "KALMAN POCKET PHONE"):

        super().__init__(
            x_inputs=NNS_INPUTS.WATCH_HIP,
            y_targets=NNS_TARGETS.ORI_CAL_LARM_UARM_HIPS,
            smooth=smooth,
            seq_len=window_size,
            stream_mc=stream_mc,
            tag=tag
        )

        self.__tag = tag
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

        self.__batch_size = 1
        self.__dim_x = 14
        self.__dim_z = 14
        self.__input_size_1 = 22
        self.__num_ensemble = num_ensemble
        self.__win_size = window_size

        self.__utils = kalman_models.Utils(
            self.__num_ensemble,
            self.__dim_x,
            self.__dim_z
        )
        self.__model = kalman_models.KalmanSmartwatchModel(
            self.__num_ensemble,
            self.__win_size,
            self.__dim_x,
            self.__dim_z,
            self.__input_size_1
        )
        self.__model.eval()
        if torch.cuda.is_available():
            self.__model.cuda()

        # Load the pretrained model
        if torch.cuda.is_available():
            checkpoint = torch.load(config.PATHS["deploy"] / "kalman" / model_hash)
            self.__model.load_state_dict(checkpoint["model"])
        else:
            checkpoint = torch.load(config.PATHS["deploy"] / "kalman" / model_hash,
                                    map_location=torch.device("cpu"))
            self.__model.load_state_dict(checkpoint["model"])

        # INIT MODEL
        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self.__init_step = 0
        self.__input_state = torch.tensor(
            np.zeros(
                (self.__batch_size, self.__num_ensemble, self.__win_size, self.__dim_x)
            ), dtype=torch.float32
        ).to(self._device)

    def reset(self):
        super().reset()
        self.__input_state = torch.tensor(
            np.zeros(
                (self.__batch_size, self.__num_ensemble, self.__win_size, self.__dim_x)
            ), dtype=torch.float32
        ).to(self._device)
        self.__init_step = 0

    def parse_row_to_xx(self, row):
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

        # estimate north quat to align Z-axis of global and android coord system
        r = ts.android_quat_to_global_no_north(sw_fwd)
        y_rot = ts.reduce_global_quat_to_y_rot(r)
        quat_north = ts.euler_to_quat(np.array([0, -y_rot, 0]))
        # calibrate watch
        sw_cal_g = ts.android_quat_to_global(sw_rot, quat_north)
        sw_6drr_cal = ts.quat_to_6drr_1x6(sw_cal_g)

        ph_fwd = np.array([
            row[self.__slp["ph_forward_w"]], row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]], row[self.__slp["ph_forward_z"]]
        ])
        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
        ])
        # the device orientations if the calib position with left arm forward is perfect
        hips_dst_g = np.array([1, 0, 0, 0])
        ph_rot_g = ts.android_quat_to_global(ph_rot, quat_north)
        ph_fwd_g = ts.android_quat_to_global(ph_fwd, quat_north)
        ph_off_g = ts.hamilton_product(ts.quat_invert(ph_fwd_g), hips_dst_g)
        ph_cal_g = ts.hamilton_product(ph_rot_g, ph_off_g)

        # hip y rotation from phone
        hips_y_rot = ts.reduce_global_quat_to_y_rot(ph_cal_g)
        hips_yrot_cal_sin = np.sin(hips_y_rot)
        hips_yrot_cal_cos = np.cos(hips_y_rot)

        # assemble the entire input vector of one time step
        return np.hstack([
            row[self.__slp["sw_dt"]],
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]],
            sw_6drr_cal,
            r_pres,
            hips_yrot_cal_sin,
            hips_yrot_cal_cos
        ])

    def make_prediction_from_row_hist(self, xx_hist: np.array):
        xx_seq = rearrange(xx_hist, "(bs seq) (en feat) -> bs seq en feat", bs=1, en=1)
        xx_seq = torch.tensor(xx_seq, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            output = self.__model(xx_seq, self.__input_state)

        # not enough history yet. Make sensor model predictions until we have a time-window worth of data
        if self.__init_step <= self.__win_size:
            # there will be no prediction in this time window
            pred_ = output[3]
            pred_ = rearrange(pred_, "bs en dim -> (bs en) dim")
            pred_ = self.__utils.format_state(pred_)
            pred_ = rearrange(
                pred_, "(bs en) (k dim) -> bs en k dim", bs=self.__batch_size, k=1
            )
            self.__input_state = torch.cat(
                (self.__input_state[:, :, 1:, :], pred_), axis=2
            ).to(self._device)
            self.__init_step += 1
            # if on GPU copy the tensor to host memory first
            if self._device.type == 'cuda':
                smp = output[3].cpu().detach().numpy()[0]
            else:
                smp = output[3].detach().numpy()[0]
            return smp  # return only sensor model prediction

        ensemble = output[0]  # -> output ensemble
        ensemble_ = rearrange(ensemble, "bs (en k) dim -> bs en k dim", k=1)
        self.__input_state = torch.cat(
            (self.__input_state[:, :, 1:, :], ensemble_), axis=2
        ).to(self._device)

        # if on GPU copy the tensor to host memory first
        if self._device.type == 'cuda':
            ensemble = ensemble.cpu()

        # get the output
        t_preds = ensemble.detach().numpy()[0]
        return t_preds

    @abstractmethod
    def process_msg(self, msg: np.array):
        return
