import logging
import socket
import struct
import time
from datetime import datetime
import queue

import numpy as np
import torch

from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.predict import models, estimate_joints
from wear_mocap_ape.utility import transformations as ts, data_stats
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class FreeHipsUDP:
    def __init__(self,
                 ip: str,
                 port: int,
                 model_hash: str,
                 smooth: int = 5,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25,
                 bonemap: BoneMap = None,
                 tag: str = "PUB FREE HIPS"):

        self.__tag = tag
        self.__port = port
        self.__ip = ip
        self.__stream_mc = stream_monte_carlo
        self.__mc_samples = monte_carlo_samples

        # average over multiple time steps
        self.__smooth = smooth
        self.__smooth_hist = []
        self.__last_hip_pred = None

        # simple lookup for values of interest
        self.__slp = messaging.WATCH_PHONE_IMU_LOOKUP

        self.__start = datetime.now()
        self.__dat = 0
        self.__prev_t = datetime.now()
        self.__row_hist = []

        # use arm length measurements for predictions
        if bonemap is None:
            # default values
            self.__larm_vec = np.array([-BoneMap.DEFAULT_LARM_LEN, 0, 0])
            self.__uarm_vec = np.array([-BoneMap.DEFAULT_UARM_LEN, 0, 0])
            self.__shou_orig = BoneMap.DEFAULT_L_SHOU_ORIG_RH

        else:
            self.__larm_vec = bonemap.left_lower_arm_vec
            self.__uarm_vec = bonemap.left_upper_arm_vec
            self.__shou_orig = bonemap.left_upper_arm_origin_rh

        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self.__body_measurements = np.repeat(
            np.r_[self.__uarm_vec, self.__larm_vec, self.__shou_orig][np.newaxis, :],
            self.__mc_samples * self.__smooth,
            axis=0
        )

        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    def row_to_pose(self, row):

        # second-wise updates to determine message frequency
        now = datetime.now()
        if (now - self.__start).seconds >= 5:
            self.__start = now
            logging.info(f"[{self.__tag}] {self.__dat / 5} Hz")
            self.__dat = 0
        delta_t = now - self.__prev_t
        delta_t = delta_t.microseconds * 0.000001
        self.__prev_t = now

        # process watch and phone data where needed
        ph_rot = np.array([
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
        ])
        ph_rot_g = ts.android_quat_to_global_no_north(ph_rot)
        ph_rot_6drr = ts.quat_to_6drr_1x6(ph_rot_g)

        sw_rot = np.array([
            row[self.__slp["sw_rotvec_w"]], row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]], row[self.__slp["sw_rotvec_z"]]
        ])
        sw_rot_g = ts.android_quat_to_global_no_north(sw_rot)
        sw_rot_6drr = ts.quat_to_6drr_1x6(sw_rot_g)

        # pressure - calibrated initial pressure = relative pressure
        r_pres = row[self.__slp["sw_pres"]] - row[self.__slp["sw_init_pres"]]

        if self.__last_hip_pred is not None:
            gt_hips_yrot_cal_sin_tm1 = self.__last_hip_pred[0]
            gt_hips_yrot_cal_cos_tm1 = self.__last_hip_pred[1]
        else:
            y_rot = ts.reduce_global_quat_to_y_rot(sw_rot_g)
            gt_hips_yrot_cal_sin_tm1 = np.sin(y_rot)
            gt_hips_yrot_cal_cos_tm1 = np.cos(y_rot)

        # assemble the entire input vector of one time step
        xx = np.hstack([
            delta_t,
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]],
            sw_rot_6drr,
            r_pres,
            row[self.__slp["ph_gyro_x"]], row[self.__slp["ph_gyro_y"]], row[self.__slp["ph_gyro_z"]],
            row[self.__slp["ph_lvel_x"]], row[self.__slp["ph_lvel_y"]], row[self.__slp["ph_lvel_z"]],
            row[self.__slp["ph_lacc_x"]], row[self.__slp["ph_lacc_y"]], row[self.__slp["ph_lacc_z"]],
            row[self.__slp["ph_grav_x"]], row[self.__slp["ph_grav_y"]], row[self.__slp["ph_grav_z"]],
            ph_rot_6drr,
            gt_hips_yrot_cal_sin_tm1,
            gt_hips_yrot_cal_cos_tm1
        ])

        if self.__normalize:
            # normalize measurements with pre-calculated mean and std
            xx = (xx - self.__xx_m) / self.__xx_s

        # sequences are used for recurrent nets. Stack time steps along 2nd axis
        self.__row_hist.append(xx)
        if len(self.__row_hist) < self.__sequence_len:
            return None

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

        # store t_preds in history if smoothing is required
        if self.__smooth > 1:
            self.__smooth_hist.append(t_preds)
            if len(self.__smooth_hist) < self.__smooth:
                return None

            while len(self.__smooth_hist) > self.__smooth:
                del self.__smooth_hist[0]

            t_preds = np.vstack(self.__smooth_hist)

        # finally, estimate hand and lower arm origins from prediction data
        est = estimate_joints.arm_pose_from_nn_targets(
            preds=t_preds,
            body_measurements=self.__body_measurements,
            y_targets=self.__y_targets
        )

        self.__last_hip_pred = [np.mean(t_preds[:, 12]), np.mean(t_preds[:, 13])]

        # estimate mean of rotations if we got multiple MC predictions
        if est.shape[0] > 1:
            # Calculate the mean of all predictions mean
            p_larm_quat_g = ts.average_quaternions(est[:, 9:13])
            p_uarm_quat_g = ts.average_quaternions(est[:, 13:17])
            p_hips_quat_g = ts.average_quaternions(est[:, 17:])

            # get the transition from upper arm origin to lower arm origin
            p_uarm_orig_rh = ts.quat_rotate_vector(p_hips_quat_g, self.__shou_orig)
            p_larm_orig_rh = ts.quat_rotate_vector(p_uarm_quat_g, self.__uarm_vec) + p_uarm_orig_rh
            p_hand_orig_rh = ts.quat_rotate_vector(p_larm_quat_g, self.__larm_vec) + p_larm_orig_rh

        else:
            p_hand_orig_rh = est[0, 0:3]
            p_larm_orig_rh = est[0, 3:6]
            p_uarm_orig_rh = est[0, 6:9]
            p_larm_quat_g = est[0, 9:13]
            p_uarm_quat_g = est[0, 13:17]
            p_hips_quat_g = est[0, 17:]

        # this is the list for the actual joint positions and rotations
        msg = [
            # hand
            p_larm_quat_g[0],
            p_larm_quat_g[1],
            p_larm_quat_g[2],
            p_larm_quat_g[3],

            p_hand_orig_rh[0],
            p_hand_orig_rh[1],
            p_hand_orig_rh[2],

            # larm
            p_larm_quat_g[0],
            p_larm_quat_g[1],
            p_larm_quat_g[2],
            p_larm_quat_g[3],

            p_larm_orig_rh[0],
            p_larm_orig_rh[1],
            p_larm_orig_rh[2],

            # uarm
            p_uarm_quat_g[0],
            p_uarm_quat_g[1],
            p_uarm_quat_g[2],
            p_uarm_quat_g[3],

            p_uarm_orig_rh[0],
            p_uarm_orig_rh[1],
            p_uarm_orig_rh[2],

            # hips
            p_hips_quat_g[0],
            p_hips_quat_g[1],
            p_hips_quat_g[2],
            p_hips_quat_g[3]
        ]

        # now we attach the monte carlo predictions for XYZ positions
        if self.__stream_mc:
            if est.shape[0] > 1:
                for e_row in est:
                    msg += list(e_row[:9])

        return msg

    def stream_loop(self, sensor_q: queue):

        logging.info("[{}] starting Unity stream loop".format(self.__tag))

        # used to estimate delta time and processing speed in Hz
        self.__start = datetime.now()
        self.__dat = 0
        self.__prev_t = datetime.now()

        # this loops while the socket is listening and/or receiving data
        while True:
            # get the most recent smartwatch data row from the queue
            row = sensor_q.get()

            while sensor_q.qsize() > 5:
                row = sensor_q.get()

            # process received data
            if row is not None:

                # get message as numpy array
                np_msg = self.row_to_pose(row)
                # can return None if not enough historical data for smoothing is available
                if np_msg is None:
                    continue

                # send message to Unity
                self.send_np_msg(msg=np_msg)
                self.__dat += 1
                time.sleep(0.01)

    def send_np_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        return self.__udp_socket.sendto(msg, (self.__ip, self.__port))
