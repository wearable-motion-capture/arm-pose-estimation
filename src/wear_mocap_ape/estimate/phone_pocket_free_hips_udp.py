import logging
import socket
import struct
import threading
import time
from datetime import datetime
import queue
from pathlib import Path

import numpy as np
import pandas as pd
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
                 smooth: int = 1,
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
            self.__uarm_orig = BoneMap.DEFAULT_L_SHOU_ORIG_RH

            self.__larm_vert = np.array([
                [-BoneMap.DEFAULT_LARM_LEN, 0.03, 0.03],
                [-BoneMap.DEFAULT_LARM_LEN, 0.03, -0.03],
                [-BoneMap.DEFAULT_LARM_LEN, -0.03, -0.03],
                [-BoneMap.DEFAULT_LARM_LEN, -0.03, 0.03],
                [0, 0.03, 0.03],
                [0, 0.03, -0.03],
                [0, -0.03, -0.03],
                [0, -0.03, 0.03],
            ])
            self.__uarm_vert = np.array([
                [-BoneMap.DEFAULT_UARM_LEN, 0.05, 0.05],
                [-BoneMap.DEFAULT_UARM_LEN, 0.05, -0.05],
                [-BoneMap.DEFAULT_UARM_LEN, -0.05, -0.05],
                [-BoneMap.DEFAULT_UARM_LEN, -0.05, 0.05],
                [0, 0.05, 0.05],
                [0, 0.05, -0.05],
                [0, -0.05, -0.05],
                [0, -0.05, 0.05],
            ])
        else:
            self.__larm_vec = bonemap.left_lower_arm_vec
            self.__uarm_vec = bonemap.left_upper_arm_vec
            self.__uarm_orig = bonemap.left_upper_arm_origin_rh

        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self.__body_measurements = np.r_[self.__larm_vec, self.__uarm_vec, self.__uarm_orig][np.newaxis, :]

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

    def parse_row_to_xx(self, row):
        # get relevant entries from the row
        sw_fwd = np.c_[
            row[self.__slp["sw_forward_w"]], row[self.__slp["sw_forward_x"]],
            row[self.__slp["sw_forward_y"]], row[self.__slp["sw_forward_z"]]
        ]

        sw_rot = np.c_[
            row[self.__slp["sw_rotvec_w"]], row[self.__slp["sw_rotvec_x"]],
            row[self.__slp["sw_rotvec_y"]], row[self.__slp["sw_rotvec_z"]]
        ]

        ph_fwd = np.c_[
            row[self.__slp["ph_forward_w"]], row[self.__slp["ph_forward_x"]],
            row[self.__slp["ph_forward_y"]], row[self.__slp["ph_forward_z"]]
        ]

        ph_rot = np.c_[
            row[self.__slp["ph_rotvec_w"]], row[self.__slp["ph_rotvec_x"]],
            row[self.__slp["ph_rotvec_y"]], row[self.__slp["ph_rotvec_z"]]
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

        # hip y rotation from phone
        hips_y_rot = ts.reduce_global_quat_to_y_rot(ph_cal_g)
        hips_yrot_cal_sin = np.sin(hips_y_rot)
        hips_yrot_cal_cos = np.cos(hips_y_rot)

        sw_rot_6drr = ts.quat_to_6drr_1x6(sw_cal_g)

        # pressure - calibrated initial pressure = relative pressure
        r_pres = row[self.__slp["sw_pres"]] - row[self.__slp["sw_init_pres"]]

        # assemble the entire input vector of one time step
        return np.c_[
            row[self.__slp["sw_dt"]],
            row[self.__slp["sw_gyro_x"]], row[self.__slp["sw_gyro_y"]], row[self.__slp["sw_gyro_z"]],
            row[self.__slp["sw_lvel_x"]], row[self.__slp["sw_lvel_y"]], row[self.__slp["sw_lvel_z"]],
            row[self.__slp["sw_lacc_x"]], row[self.__slp["sw_lacc_y"]], row[self.__slp["sw_lacc_z"]],
            row[self.__slp["sw_grav_x"]], row[self.__slp["sw_grav_y"]], row[self.__slp["sw_grav_z"]],
            sw_rot_6drr,
            r_pres,
            hips_yrot_cal_sin,
            hips_yrot_cal_cos
        ]

    def stream_loop(self, sensor_q: queue):
        logging.info("[{}] starting Unity stream loop".format(self.__tag))
        # used to estimate delta time and processing speed in Hz
        self.__start = datetime.now()
        self.__dat = 0
        self.__prev_t = datetime.now()
        self.__row_hist = []
        # this loops while the socket is listening and/or receiving data
        while True:
            # get the most recent smartwatch data row from the queue
            xx = self.parse_row_to_xx(sensor_q.get())

            # add rows until the queue is nearly empty
            while sensor_q.qsize() > 5:
                self.__row_hist.append(xx)  # add row to history and immediately continue with the next
                xx = self.parse_row_to_xx(sensor_q.get())

            # get message as numpy array
            est = self.make_prediction(xx)

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

            # now we attach the monte carlo predictions for XYZ positions
            if self.__stream_mc:
                if est.shape[0] > 1:
                    msg = np.hstack([msg, est[:, :9].flatten()])

            np_msg = np.array(msg)
            self.last_msg = np_msg

            # five-secondly updates to log message frequency
            now = datetime.now()
            if (now - self.__start).seconds >= 5:
                self.__start = now
                logging.info(f"[{self.__tag}] {self.__dat / 5} Hz")
                self.__dat = 0

            # send message to Unity
            self.send_np_msg(msg=np_msg)
            self.__dat += 1
            time.sleep(0.01)

    def errors_from_file(self, file_p: Path, publish: bool = False):
        logging.info(f"[{self.__tag}] processing from file")
        hand_errors, elbow_errors, hip_rot_errors, mpjve = [], [], [], []

        dat = pd.read_csv(file_p)
        for i, row in dat.iterrows():

            ## PREDICTIONS
            xx = row[self.__x_inputs.value].to_numpy()
            # normalize measurements with pre-calculated mean and std
            est = self.make_prediction(xx)

            # estimate mean of rotations if we got multiple MC predictions
            if est.shape[0] > 1:
                # Calculate the mean of all predictions mean
                p_larm_quat_rh = ts.average_quaternions(est[:, 9:13])
                p_uarm_quat_rh = ts.average_quaternions(est[:, 13:17])
                p_hips_quat_rh = ts.average_quaternions(est[:, 17:])

                # get the transition from upper arm origin to lower arm origin
                p_uarm_orig_rh = ts.quat_rotate_vector(p_hips_quat_rh, self.__uarm_orig)
                p_larm_orig_rh = ts.quat_rotate_vector(p_uarm_quat_rh, self.__uarm_vec) + p_uarm_orig_rh
                p_hand_orig_rh = ts.quat_rotate_vector(p_larm_quat_rh, self.__larm_vec) + p_larm_orig_rh
            else:
                p_hand_orig_rh = est[0, 0:3]
                p_larm_orig_rh = est[0, 3:6]
                p_uarm_orig_rh = est[0, 6:9]
                p_larm_quat_rh = est[0, 9:13]
                p_uarm_quat_rh = est[0, 13:17]
                p_hips_quat_rh = est[0, 17:]

            # publish the estimation for our unity visualization
            if publish:
                msg = np.hstack([
                    p_larm_quat_rh,
                    p_hand_orig_rh,
                    p_larm_quat_rh,
                    p_larm_orig_rh,
                    p_uarm_quat_rh,
                    p_uarm_orig_rh,
                    p_hips_quat_rh
                ])
                if est.shape[0] > 1:
                    if est.shape[0] > 1:
                        msg = np.hstack([msg, est[:, :9].flatten()])

                # five-secondly updates to log message frequency
                now = datetime.now()
                if (now - self.__start).seconds >= 5:
                    self.__start = now
                    logging.info(f"[{self.__tag}] {self.__dat / 5} Hz")
                    self.__dat = 0

                # send message to Unity
                self.send_np_msg(msg=msg)
                self.__dat += 1

            # GROUND TRUTH
            yy = row[self.__y_targets.value].to_numpy()[np.newaxis, :]
            est = estimate_joints.arm_pose_from_nn_targets(
                preds=yy,
                body_measurements=self.__body_measurements,
                y_targets=self.__y_targets
            )
            gt_hand_orig_rh = est[0, 0:3]
            gt_larm_orig_rh = est[0, 3:6]
            gt_uarm_orig_rh = est[0, 6:9]
            gt_larm_quat_g = est[0, 9:13]
            gt_uarm_quat_g = est[0, 13:17]
            gt_hips_quat_g = est[0, 17:]

            p_larm_vrtx = ts.quat_rotate_vector(p_larm_quat_rh, self.__larm_vert) + p_larm_orig_rh
            p_uarm_vrtx = ts.quat_rotate_vector(p_uarm_quat_rh, self.__uarm_vert) + p_uarm_orig_rh
            gt_larm_vrtx = ts.quat_rotate_vector(gt_larm_quat_g, self.__larm_vert) + gt_larm_orig_rh
            gt_uarm_vrtx = ts.quat_rotate_vector(gt_uarm_quat_g, self.__uarm_vert) + gt_uarm_orig_rh

            le = gt_larm_vrtx - p_larm_vrtx
            ue = gt_uarm_vrtx - p_uarm_vrtx
            ae = np.vstack([le, ue])
            me = np.linalg.norm(ae, axis=1)

            mpjve.append(np.mean(me) * 100)
            hand_errors.append(np.linalg.norm(gt_hand_orig_rh - p_hand_orig_rh) * 100)
            elbow_errors.append(np.linalg.norm(gt_larm_orig_rh - p_larm_orig_rh) * 100)
            hip_rot_errors.append(
                np.degrees(np.abs(ts.quat_to_euler(gt_hips_quat_g)[1] - ts.quat_to_euler(p_hips_quat_rh)[1]))
            )

            if (int(i) + 1) % 100 == 0:
                logging.info(f"[{self.__tag}] processed {i} rows")

        return hand_errors, elbow_errors, hip_rot_errors, mpjve

    def send_np_msg(self, msg: np.array) -> int:
        # craft UDP message and send
        msg = struct.pack('f' * len(msg), *msg)
        return self.__udp_socket.sendto(msg, (self.__ip, self.__port))

    def make_prediction(self, xx):

        self.__row_hist.append(xx)

        # if not enough data is available yet, simply repeat the input as a first estimate
        while len(self.__row_hist) < self.__sequence_len:
            self.__row_hist.append(xx)

        # if the history is too long, delete old values
        while len(self.__row_hist) > self.__sequence_len:
            del self.__row_hist[0]

        # stack rows to one big array
        seq = np.vstack(self.__row_hist.copy())

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

        est = estimate_joints.arm_pose_from_nn_targets(
            preds=t_preds,
            body_measurements=self.__body_measurements,
            y_targets=self.__y_targets
        )

        # store t_preds in history if smoothing is required
        if self.__smooth > 1:
            self.__smooth_hist.append(est)
            while len(self.__smooth_hist) < self.__smooth:
                self.__smooth_hist.append(est)
            while len(self.__smooth_hist) > self.__smooth:
                del self.__smooth_hist[0]
            est = np.vstack(self.__smooth_hist)

        return est


def run_free_hips_pocket_udp(ip: str,
                             model_hash: str = deploy_models.LSTM.POCKET_MODE.value,
                             stream_monte_carlo: bool = False) -> FreeHipsPocketUDP:
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
                            smooth=10,
                            port=config.PORT_PUB_LEFT_ARM,
                            monte_carlo_samples=25,
                            stream_monte_carlo=stream_monte_carlo)
    p_thread = threading.Thread(
        target=fhp.stream_loop,
        args=(left_q,)
    )
    p_thread.start()

    return fhp
