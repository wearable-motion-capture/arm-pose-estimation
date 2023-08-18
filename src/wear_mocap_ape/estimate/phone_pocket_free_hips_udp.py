import logging
import socket
import struct
import threading
import time
from datetime import datetime
import queue

import numpy as np
import torch

from wear_mocap_ape import config
from wear_mocap_ape.data_deploy.nn import deploy_models
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate import models, estimate_joints
from wear_mocap_ape.stream.listener.imu import ImuListener
from wear_mocap_ape.utility import transformations as ts, data_stats
from wear_mocap_ape.data_types import messaging
from wear_mocap_ape.utility.names import NNS_TARGETS, NNS_INPUTS


class FreeHipsPocketUDP:
    def __init__(self,
                 ip: str,
                 port: int,
                 model_hash: str,
                 smooth: int = 5,
                 stream_monte_carlo=True,
                 monte_carlo_samples=25,
                 bonemap: BoneMap = None,
                 tag: str = "PUB FREE HIPS"):

        # smooth should not be smaller 1
        smooth = max(1, smooth)

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

        self.last_msg = np.zeros(25)

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

    @property
    def sequence_len(self):
        return self.__sequence_len

    def calibrate_imus_with_offset(self, seq: np.array):
        # get relevant entries from the row
        sw_fwd = np.c_[
            seq[:, self.__slp["sw_forward_w"]], seq[:, self.__slp["sw_forward_x"]],
            seq[:, self.__slp["sw_forward_y"]], seq[:, self.__slp["sw_forward_z"]]
        ]

        sw_rot = np.c_[
            seq[:, self.__slp["sw_rotvec_w"]], seq[:, self.__slp["sw_rotvec_x"]],
            seq[:, self.__slp["sw_rotvec_y"]], seq[:, self.__slp["sw_rotvec_z"]]
        ]

        ph_fwd = np.c_[
            seq[:, self.__slp["ph_forward_w"]], seq[:, self.__slp["ph_forward_x"]],
            seq[:, self.__slp["ph_forward_y"]], seq[:, self.__slp["ph_forward_z"]]
        ]

        ph_rot = np.c_[
            seq[:, self.__slp["ph_rotvec_w"]], seq[:, self.__slp["ph_rotvec_x"]],
            seq[:, self.__slp["ph_rotvec_y"]], seq[:, self.__slp["ph_rotvec_z"]]
        ]

        # estimate north quat to align Z-axis of global and android coord system
        r = ts.android_quat_to_global_no_north(sw_fwd)
        y_rot = ts.reduce_global_quat_to_y_rot(r)
        quat_north = ts.euler_to_quat(np.c_[np.zeros(y_rot.shape), -y_rot, np.zeros(y_rot.shape)])

        # calibrate watch
        sw_cal_g = ts.android_quat_to_global(sw_rot, quat_north)

        # the device orientations if the calib position with left arm forward is perfect
        hips_dst_g = np.array([1, 0, 0, 0])
        ph_rot_g = ts.android_quat_to_global(ph_rot, quat_north)
        ph_fwd_g = ts.android_quat_to_global(ph_fwd, quat_north)
        ph_off_g = ts.hamilton_product(ts.quat_invert(ph_fwd_g), hips_dst_g)
        ph_cal_g = ts.hamilton_product(ph_rot_g, ph_off_g)

        return sw_cal_g, ph_cal_g

    def row_hist_to_pose(self, row_hist):

        # stack rows to one big array
        seq = np.vstack(row_hist)

        sw_cal_g, ph_cal_g = self.calibrate_imus_with_offset(seq)

        # hip y rotation from phone
        hips_y_rot = ts.reduce_global_quat_to_y_rot(ph_cal_g)
        hips_quat_g = ts.euler_to_quat(np.c_[np.zeros(hips_y_rot.shape), hips_y_rot, np.zeros(hips_y_rot.shape)])
        hips_yrot_cal_sin = np.sin(hips_y_rot)
        hips_yrot_cal_cos = np.cos(hips_y_rot)

        # relative smartwatch orientation
        sw_cal_rh = ts.hamilton_product(ts.quat_invert(hips_quat_g), sw_cal_g)
        sw_rot_6drr = ts.quat_to_6drr_1x6(sw_cal_rh)

        # pressure - calibrated initial pressure = relative pressure
        r_pres = seq[:, self.__slp["sw_pres"]] - seq[:, self.__slp["sw_init_pres"]]

        # assemble the entire input vector of one time step
        xx = np.c_[
            seq[:, self.__slp["sw_dt"]],
            seq[:, self.__slp["sw_gyro_x"]], seq[:, self.__slp["sw_gyro_y"]], seq[:, self.__slp["sw_gyro_z"]],
            seq[:, self.__slp["sw_lvel_x"]], seq[:, self.__slp["sw_lvel_y"]], seq[:, self.__slp["sw_lvel_z"]],
            seq[:, self.__slp["sw_lacc_x"]], seq[:, self.__slp["sw_lacc_y"]], seq[:, self.__slp["sw_lacc_z"]],
            seq[:, self.__slp["sw_grav_x"]], seq[:, self.__slp["sw_grav_y"]], seq[:, self.__slp["sw_grav_z"]],
            sw_rot_6drr,
            r_pres
        ]

        if self.__normalize:
            # normalize measurements with pre-calculated mean and std
            xx = (xx - self.__xx_m) / self.__xx_s

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

        t_preds = np.c_[
            t_preds,
            np.repeat(hips_yrot_cal_sin[-1], t_preds.shape[0]),
            np.repeat(hips_yrot_cal_cos[-1], t_preds.shape[0])
        ]

        # finally, estimate hand and lower arm origins from prediction data
        if self.__y_targets == NNS_TARGETS.POS_RH_LARM_HAND:
            hip_targets = NNS_TARGETS.POS_RH_LARM_HAND_HIPS
        else:
            hip_targets = NNS_TARGETS.ORI_CALIB_UARM_LARM_HIPS

        est = estimate_joints.arm_pose_from_nn_targets(
            preds=t_preds,
            body_measurements=self.__body_measurements,
            y_targets=hip_targets
        )

        # estimate mean of rotations if we got multiple MC predictions
        if est.shape[0] > 1:
            # Calculate the mean of all predictions mean
            p_hips_quat_g = ts.average_quaternions(est[:, 17:])
            p_larm_quat_g = ts.average_quaternions(est[:, 9:13])
            p_uarm_quat_g = ts.average_quaternions(est[:, 13:17])

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

        row_hist = []

        # this loops while the socket is listening and/or receiving data
        while True:
            # get the most recent smartwatch data row from the queue
            row_hist.append(sensor_q.get())

            # add rows until the queue is nearly empty
            while sensor_q.qsize() > 5:
                row_hist.append(sensor_q.get())

            # only proceed if the history is long enough
            if len(row_hist) < self.__sequence_len:
                continue

            # if the history is too long, delete old values
            while len(row_hist) > self.__sequence_len:
                del row_hist[0]

            # get message as numpy array
            np_msg = self.row_hist_to_pose(row_hist)

            # can return None if not enough historical data for smoothing is available
            if np_msg is None:
                continue

            self.last_msg = np_msg

            # five-secondly updates to log message frequency
            now = datetime.now()
            if (now - self.__start).seconds >= 5:
                self.__start = now
                logging.info(f"[{self.__tag}] {self.__dat / 5} Hz")
                self.__dat = 0
            # delta_t = now - self.__prev_t
            # delta_t = delta_t.microseconds * 0.000001
            # self.__prev_t = now

            # send message to Unity
            self.send_np_msg(msg=np_msg)
            self.__dat += 1
            time.sleep(0.01)

    def send_np_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        return self.__udp_socket.sendto(msg, (self.__ip, self.__port))


def run_free_hips_pocket_udp(ip: str, stream_monte_carlo: bool = False) -> FreeHipsPocketUDP:
    model_hash = deploy_models.LSTM.WATCH_ONLY.value

    # data for left-hand mode
    left_q = queue.Queue()

    # listen for imu data from phone and watch
    imu_l = ImuListener(
        ip=ip,
        msg_size=messaging.watch_phone_imu_msg_len,
        port=config.PORT_LISTEN_WATCH_PHONE_IMU_LEFT,
        tag="LISTEN IMU LEFT"
    )
    l_thread = threading.Thread(
        target=imu_l.listen,
        args=(left_q,)
    )
    l_thread.start()

    # process into arm pose and body orientation
    fhp = FreeHipsPocketUDP(ip=ip,
                            model_hash=model_hash,
                            port=config.PORT_PUB_LEFT_ARM,
                            stream_monte_carlo=stream_monte_carlo)
    p_thread = threading.Thread(
        target=fhp.stream_loop,
        args=(left_q,)
    )
    p_thread.start()

    return fhp
