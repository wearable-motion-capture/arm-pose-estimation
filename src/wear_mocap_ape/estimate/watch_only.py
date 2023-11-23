import logging
from abc import abstractmethod
from datetime import datetime

import torch
import queue
import numpy as np

from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.estimate import estimate_joints, models
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.utility import transformations as ts, data_stats
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class WatchOnly:
    def __init__(self,
                 model_hash: str = deploy_models.LSTM.WATCH_ONLY.value,
                 smooth: int = 10,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25,
                 bonemap: BoneMap = None,
                 tag: str = "PUB WATCH"):

        self.__tag = tag
        self.__active = True

        # average over multiple time steps
        self.__smooth = smooth
        self.__smooth_hist = []

        # monte carlo predictions
        self.__stream_mc = stream_monte_carlo
        self.__mc_samples = monte_carlo_samples
        self.__last_msg = None

        self.__row_hist = []

        # use body measurements for transitions
        if bonemap is None:
            self.__larm_vec = np.array([-BoneMap.DEFAULT_LARM_LEN, 0, 0])
            self.__uarm_vec = np.array([-BoneMap.DEFAULT_UARM_LEN, 0, 0])
            self.__uarm_orig = BoneMap.DEFAULT_L_SHOU_ORIG_RH
        else:
            # get values from bone map
            self.__larm_vec = np.array([-bonemap.left_lower_arm_length, 0, 0])
            self.__uarm_vec = np.array([-bonemap.left_upper_arm_length, 0, 0])
            self.__uarm_orig = bonemap.left_upper_arm_origin_rh
        self.__hips_quat = np.array([1, 0, 0, 0])

        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self.__body_measurements = np.r_[self.__uarm_vec, self.__larm_vec, self.__uarm_orig][np.newaxis, :]

        # load the trained network
        torch.set_default_dtype(torch.float64)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model from given parameters
        self.__nn_model, params = models.load_deployed_model_from_hash(hash_str=model_hash)
        self.__y_targets = NNS_TARGETS(params["y_targets_v"])
        self.__sequence_len = params["sequence_len"]
        self.__normalize = params["normalize"] if "normalize" in params else True

        # load normalized data stats if required
        if self.__normalize:
            stats = data_stats.get_norm_stats(x_inputs=NNS_INPUTS(params["x_inputs_v"]), y_targets=self.__y_targets)
            # data is normalized and has to be transformed with pre-calculated mean and std
            self.__xx_m, self.__xx_s = stats["xx_m"], stats["xx_s"]
            self.__yy_m, self.__yy_s = stats["yy_m"], stats["yy_s"]
        else:
            self.__xx_m, self.__xx_s = 0., 1.
            self.__yy_m, self.__yy_s = 0., 1.

    def get_last_msg(self):
        return self.__last_msg

    def is_active(self):
        return self.__active

    def terminate(self):
        self.__active = False

    def add_obs_and_make_prediction(self, xx):
        if self.__normalize:
            # normalize measurements with pre-calculated mean and std
            xx = (xx - self.__xx_m) / self.__xx_s

        self.__row_hist.append(xx)
        # if not enough data is available yet, simply repeat the input as a first estimate
        while len(self.__row_hist) < self.__sequence_len:
            self.__row_hist.append(xx)
        # if the history is too long, delete old values
        while len(self.__row_hist) > self.__sequence_len:
            del self.__row_hist[0]

        xx = np.vstack(self.__row_hist)
        # finally, cast to a torch tensor with batch size 1
        xx = torch.tensor(xx[None, :, :])
        with torch.no_grad():
            # make mote carlo predictions if the model makes use of dropout
            t_preds = self.__nn_model.monte_carlo_predictions(x=xx, n_samples=self.__mc_samples)

        # if on GPU copy the tensor to host memory first
        if self.__device.type == 'cuda':
            t_preds = t_preds.cpu()
        t_preds = t_preds.numpy()

        # we are only interested in the last prediction of the sequence
        t_preds = t_preds[:, -1, :]

        if self.__normalize:
            # transform predictions back from normalized labels
            t_preds = t_preds * self.__yy_s + self.__yy_m

        return t_preds

    def msg_from_pred(self, pred):
        est = estimate_joints.arm_pose_from_nn_targets(
            preds=pred,
            body_measurements=self.__body_measurements,
            y_targets=self.__y_targets
        )

        # store est in history if smoothing is required
        if self.__smooth > 1:
            self.__smooth_hist.append(est)
            while len(self.__smooth_hist) < self.__smooth:
                self.__smooth_hist.append(est)
            while len(self.__smooth_hist) > self.__smooth:
                del self.__smooth_hist[0]
            est = np.vstack(self.__smooth_hist)

        # estimate mean of rotations if we got multiple MC predictions
        if est.shape[0] > 1:
            # Calculate the mean of all predictions mean
            pred_larm_rot_rh = ts.average_quaternions(est[:, 6:10])
            pred_uarm_rot_rh = ts.average_quaternions(est[:, 10:])
            # use body measurements for transitions
            # get the transition from upper arm origin to lower arm origin
            pred_larm_origin_rh = ts.quat_rotate_vector(pred_uarm_rot_rh, self.__uarm_vec) + self.__uarm_orig
            # get transitions from lower arm origin to hand
            rotated_lower_arms_re = ts.quat_rotate_vector(pred_larm_rot_rh, self.__larm_vec)
            pred_hand_origin_rh = rotated_lower_arms_re + pred_larm_origin_rh
        else:
            pred_hand_origin_rh = est[0, :3]
            pred_larm_origin_rh = est[0, 3:6]
            pred_larm_rot_rh = est[0, 6:10]
            pred_uarm_rot_rh = est[0, 10:]

        # this is the list for the actual joint positions and rotations
        msg = np.hstack([
            pred_larm_rot_rh,  # hand quat
            pred_hand_origin_rh,  # hand pos
            pred_larm_rot_rh,  # larm quat
            pred_larm_origin_rh,
            pred_uarm_rot_rh,  # uarm quat
            self.__uarm_orig,
            self.__hips_quat
        ])
        # store as last msg for getter
        self.__last_msg = msg.copy()

        if self.__stream_mc:
            msg = list(msg)
            # now we attach the monte carlo predictions for XYZ positions
            if est.shape[0] > 1:
                for e_row in est:
                    msg += list(e_row[:6])
        return msg

    def processing_loop(self, sensor_q: queue):
        self.__active = True
        logging.info(f"[{self.__tag}] starting watch standalone processing loop")

        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        prev_time = datetime.now()
        dat = 0

        # simple lookup for values of interest
        slp = messaging.WATCH_ONLY_IMU_LOOKUP

        # this loops while the socket is listening and/or receiving data
        while self.__active:

            # get the most recent smartwatch data row from the queue
            row = None
            while not sensor_q.empty():
                row = sensor_q.get()

            # process received data
            if row is not None:
                # second-wise updates to determine message frequency
                now = datetime.now()
                if (now - start).seconds >= 5:
                    start = now
                    logging.info(f"[{self.__tag}] {dat / 5} Hz")
                    dat = 0
                delta_t = now - prev_time
                delta_t = delta_t.microseconds * 0.000001
                prev_time = now

                # process the data
                # pressure - calibrated initial pressure = relative pressure
                r_pres = row[slp["sw_pres"]] - row[slp["sw_init_pres"]]

                # calibrate smartwatch rotation
                sw_rot = np.array([
                    row[slp["sw_rotvec_w"]],
                    row[slp["sw_rotvec_x"]],
                    row[slp["sw_rotvec_y"]],
                    row[slp["sw_rotvec_z"]]
                ])
                sw_fwd = np.array([
                    row[slp["sw_forward_w"]],
                    row[slp["sw_forward_x"]],
                    row[slp["sw_forward_y"]],
                    row[slp["sw_forward_z"]]
                ])
                quat_north = ts.calib_watch_left_to_north_quat(sw_fwd)
                sw_quat_cal = ts.android_quat_to_global(sw_rot, quat_north)
                sw_6drr_cal = ts.quat_to_6drr_1x6(sw_quat_cal)  # get 6dof rotation representation

                # assemble the entire input vector of one time step
                xx = np.hstack([
                    delta_t,
                    row[slp["sw_gyro_x"]], row[slp["sw_gyro_y"]], row[slp["sw_gyro_z"]],
                    row[slp["sw_lvel_x"]], row[slp["sw_lvel_y"]], row[slp["sw_lvel_z"]],
                    row[slp["sw_lacc_x"]], row[slp["sw_lacc_y"]], row[slp["sw_lacc_z"]],
                    row[slp["sw_grav_x"]], row[slp["sw_grav_y"]], row[slp["sw_grav_z"]],
                    sw_6drr_cal,
                    r_pres
                ])

                t_preds = self.add_obs_and_make_prediction(xx)
                msg = self.msg_from_pred(t_preds)

                # store as last msg for getter
                self.__last_msg = msg.copy()

                # craft UDP message and send
                self.process_msg(msg=msg)
                dat += 1

    @abstractmethod
    def process_msg(self, msg: np.array):
        return
