import logging

from abc import abstractmethod
from datetime import datetime
import queue
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate import models, estimate_joints
from wear_mocap_ape.utility import transformations as ts, data_stats
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class WatchPhonePocketNN:
    def __init__(self,
                 model_hash: str,
                 smooth: int = 1,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25,
                 bonemap: BoneMap = None,
                 tag: str = "NN POCKET PHONE"):

        # smooth should not be smaller 1
        smooth = max(1, smooth)
        self.__smooth = smooth
        self.__smooth_hist = []

        self.__tag = tag
        self.__active = False

        self.__stream_mc = stream_monte_carlo
        self.__mc_samples = monte_carlo_samples
        self.last_msg = None
        self.__row_hist = []

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

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

        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self.__body_measurements = np.r_[self.__larm_vec, self.__uarm_vec, self.__uarm_orig][np.newaxis, :]

        # load the trained network
        torch.set_default_dtype(torch.float64)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model from given parameters
        self.__nn_model, params = models.load_deployed_model_from_hash(hash_str=model_hash)
        self.__y_targets = NNS_TARGETS(params["y_targets_v"])
        self.__x_inputs = NNS_INPUTS(params["x_inputs_v"])
        self.__sequence_len = params["sequence_len"]
        self.__normalize = params["normalize"]

        # load normalized data stats if required
        if self.__normalize:
            # data is normalized and has to be transformed with pre-calculated mean and std
            stats = data_stats.get_norm_stats(x_inputs=self.__x_inputs, y_targets=self.__y_targets)
            self.__xx_m, self.__xx_s = stats["xx_m"], stats["xx_s"]
            self.__yy_m, self.__yy_s = stats["yy_m"], stats["yy_s"]
        else:
            self.__xx_m, self.__xx_s = 0., 1.
            self.__yy_m, self.__yy_s = 0., 1.

    def terminate(self):
        self.__active = False

    def processing_loop(self, sensor_q: queue):
        logging.info("[{}] starting Unity stream loop".format(self.__tag))

        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        dat = 0

        # this loops while the socket is listening and/or receiving data
        self.__active = True
        while self.__active:
            # processing speed output
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{self.__tag}] {dat / 5} Hz")
                dat = 0

            # get the most recent smartwatch data row from the queue
            row = sensor_q.get()
            while sensor_q.qsize() > 5:
                row = sensor_q.get()

            # finally get predicted positions etc
            xx = self.parse_row_to_xx(row)
            t_pred = self.add_obs_and_make_prediction(xx)
            msg = self.msg_from_pred(t_pred)
            self.process_msg(msg)
            dat += 1

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

    def add_obs_and_make_prediction(self, x):
        self.__row_hist.append(x)
        # if not enough data is available yet, simply repeat the input as a first estimate
        while len(self.__row_hist) < self.__sequence_len:
            self.__row_hist.append(x)
        # if the history is too long, delete old values
        while len(self.__row_hist) > self.__sequence_len:
            del self.__row_hist[0]

        t_preds = self.make_prediction_from_row_hist(np.vstack(self.__row_hist))
        return t_preds

    def make_prediction_from_row_hist(self, row_hist):
        # stack rows to one big array
        seq = np.vstack(row_hist.copy())

        if self.__normalize:
            # normalize measurements with pre-calculated mean and std
            xx = (seq - self.__xx_m) / self.__xx_s

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
            p_hips_quat_g = ts.average_quaternions(est[:, 17:])
            p_larm_quat_g = ts.average_quaternions(est[:, 9:13])
            p_uarm_quat_g = ts.average_quaternions(est[:, 13:17])

            # get the transition from upper arm origin to lower arm origin
            p_uarm_orig_g = ts.quat_rotate_vector(p_hips_quat_g, self.__uarm_orig)
            p_larm_orig_g = ts.quat_rotate_vector(p_uarm_quat_g, self.__uarm_vec) + p_uarm_orig_g
            p_hand_orig_g = ts.quat_rotate_vector(p_larm_quat_g, self.__larm_vec) + p_larm_orig_g
        else:
            p_hand_orig_g = est[0, 0:3]
            p_larm_orig_g = est[0, 3:6]
            p_uarm_orig_g = est[0, 6:9]
            p_larm_quat_g = est[0, 9:13]
            p_uarm_quat_g = est[0, 13:17]
            p_hips_quat_g = est[0, 17:]

        # this is the list for the actual joint positions and rotations
        msg = np.hstack([
            p_larm_quat_g,  # hand rot
            p_hand_orig_g,  # hand orig
            p_larm_quat_g,  # larm rot
            p_larm_orig_g,  # larm orig
            p_uarm_quat_g,  # uarm rot
            p_uarm_orig_g,  # uarm orig
            p_hips_quat_g  # hips rot
        ])
        # store as last msg for getter
        self.last_msg = msg.copy()

        if self.__stream_mc:
            msg = list(msg)
            # now we attach the monte carlo predictions for XYZ positions
            if est.shape[0] > 1:
                for e_row in est:
                    msg += list(e_row[:6])
        return msg

    @abstractmethod
    def process_msg(self, msg: np.array):
        return

    @property
    def sequence_len(self):
        return self.__sequence_len

    @property
    def x_inputs(self):
        return self.__x_inputs

    @property
    def y_targets(self):
        return self.__y_targets

    @property
    def body_measurements(self):
        return self.__body_measurements

    @property
    def uarm_orig(self):
        return self.__uarm_orig

    @property
    def uarm_vec(self):
        return self.__uarm_vec

    @property
    def larm_vec(self):
        return self.__larm_vec
